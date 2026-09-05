from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidenceV4:
    family: str
    scores: dict
    linear_gain: float
    quadratic_gain: float
    phase_jump_strength: float
    frequency_step_strength: float
    residual_rms_rad: float


class PhysicsWaveformFrontendV4:
    """Expanded physics-based waveform family classifier.

    Broad families:
      CW             approximately constant instantaneous frequency
      LFM            approximately linear instantaneous frequency
      NLFM           smooth nonlinear instantaneous frequency
      FREQUENCY_CODED discrete frequency states/steps
      PHASE_CODED    sparse phase discontinuities with otherwise smooth carrier
      UNKNOWN        no model has adequate evidence

    This is intentionally a broad-family detector. Barker/Frank subtype analysis is
    left to later stages.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.sample_rate_hz = float(sample_rate_hz)

    @staticmethod
    def _robust_scale(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return 1e-4
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return max(1.4826 * mad, 1e-4)

    @staticmethod
    def _wrapped_residual(observed, fitted):
        return np.angle(np.exp(1j * (observed - fitted)))

    def classify(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 20:
            return PhysicsWaveformEvidenceV4("UNKNOWN", {}, 0.0, 0.0, 0.0, 0.0, float("inf"))

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 16:
            x = x[edge:-edge]

        q = np.angle(x[1:] * np.conj(x[:-1]))
        n = np.arange(len(q), dtype=float)
        if len(q) < 12:
            return PhysicsWaveformEvidenceV4("UNKNOWN", {}, 0.0, 0.0, 0.0, 0.0, float("inf"))

        # Initial robust constant model and phase-jump detection.
        q_med = float(np.median(q))
        r0 = self._wrapped_residual(q, q_med)
        abs_r0 = np.abs(r0)
        central = r0[abs_r0 <= np.quantile(abs_r0, 0.70)]
        noise = self._robust_scale(central if len(central) >= 8 else r0)

        jump_threshold = max(5.0 * noise, 0.35)
        jump_mask = abs_r0 > jump_threshold
        jump_excess = np.maximum(abs_r0[jump_mask] - jump_threshold, 0.0)
        if len(jump_excess):
            strongest = np.sort(jump_excess)[-min(12, len(jump_excess)):]
            phase_jump_strength = float(np.sum(strongest) / max(noise, 0.05))
        else:
            phase_jump_strength = 0.0
        phase_score = float(1.0 - np.exp(-phase_jump_strength / 16.0))

        # Smooth samples are used for polynomial FM model comparisons.
        keep = ~jump_mask
        if np.count_nonzero(keep) < 10:
            keep = np.ones_like(jump_mask, dtype=bool)
        ns = n[keep]
        qs = q[keep]
        nn = (ns - np.mean(ns)) / max(np.std(ns), 1.0)

        # Constant, linear, quadratic phase-increment models.
        a0 = np.ones((len(nn), 1))
        a1 = np.column_stack((np.ones_like(nn), nn))
        a2 = np.column_stack((np.ones_like(nn), nn, nn ** 2))

        def fit_sse(a):
            coeff, _, _, _ = np.linalg.lstsq(a, qs, rcond=None)
            resid = self._wrapped_residual(qs, a @ coeff)
            return float(np.sum(resid ** 2)), float(np.sqrt(np.mean(resid ** 2)))

        sse0, rms0 = fit_sse(a0)
        sse1, rms1 = fit_sse(a1)
        sse2, rms2 = fit_sse(a2)

        linear_gain = max(0.0, sse0 - sse1) / max(sse0, 1e-12) * max(len(qs) - 2, 1)
        quadratic_gain = max(0.0, sse1 - sse2) / max(sse1, 1e-12) * max(len(qs) - 3, 1)

        cw_score = float(1.0 / (1.0 + linear_gain / 3.0))
        lfm_score = float((linear_gain / (linear_gain + 4.0)) * (1.0 / (1.0 + quadratic_gain / 5.0)))
        nlfm_score = float((linear_gain / (linear_gain + 3.0)) * (quadratic_gain / (quadratic_gain + 4.0)))

        # Detect frequency coding as repeated piecewise-constant phase increments.
        # Quantize the instantaneous-frequency proxy to a noise-scaled grid and
        # measure both state concentration and transition sparsity.
        quant = max(3.0 * noise, 0.03)
        states = np.round(q / quant).astype(int)
        if len(states) > 1:
            transitions = np.count_nonzero(states[1:] != states[:-1])
            transition_fraction = transitions / (len(states) - 1)
        else:
            transition_fraction = 1.0
        unique, counts = np.unique(states, return_counts=True)
        dominant_mass = float(np.sum(np.sort(counts)[-min(6, len(counts)):]) / len(states))
        discrete_structure = dominant_mass * max(0.0, 1.0 - transition_fraction)
        frequency_step_strength = float(discrete_structure)
        freq_code_score = float(np.clip((discrete_structure - 0.35) / 0.45, 0.0, 1.0))

        # Phase-code evidence should suppress smooth CW/LFM/NLFM interpretations.
        smooth_suppression = 1.0 - 0.80 * phase_score
        cw_score *= smooth_suppression
        lfm_score *= smooth_suppression
        nlfm_score *= smooth_suppression

        # Frequency-coded signals often also fit poor smooth models; prefer the
        # discrete-state explanation when its structure is strong.
        scores = {
            "CW": cw_score,
            "LFM": lfm_score,
            "NLFM": nlfm_score,
            "FREQUENCY_CODED": freq_code_score,
            "PHASE_CODED": phase_score,
        }

        family = max(scores, key=scores.get)
        best = scores[family]
        ranked = sorted(scores.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]

        # UNKNOWN is reserved for cases where no broad physical family gives
        # adequate evidence, or where the competition is weak and ambiguous.
        if best < 0.22 or (best < 0.40 and margin < 0.04):
            family = "UNKNOWN"

        return PhysicsWaveformEvidenceV4(
            family=family,
            scores=scores,
            linear_gain=float(linear_gain),
            quadratic_gain=float(quadratic_gain),
            phase_jump_strength=float(phase_jump_strength),
            frequency_step_strength=float(frequency_step_strength),
            residual_rms_rad=float(rms2),
        )
