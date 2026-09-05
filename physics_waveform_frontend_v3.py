from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidenceV3:
    coarse_family: str
    cw_score: float
    lfm_score: float
    phase_coded_score: float
    frequency_slope_hz_per_s: float
    smooth_residual_rad: float
    jump_strength: float
    linear_model_gain: float


class PhysicsWaveformFrontendV3:
    """V3 physics front end.

    Phase-code detection is intentionally retained from V2. The CW/LFM decision is
    changed to a nested-model comparison: constant differential phase (CW) versus
    linear differential phase (LFM). The linear model must reduce residual energy
    enough to justify its extra slope parameter.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.sample_rate_hz = float(sample_rate_hz)

    @staticmethod
    def _robust_scale(values):
        values = np.asarray(values, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(1.4826 * mad, 1e-4)

    def classify(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 16:
            return PhysicsWaveformEvidenceV3("UNKNOWN", 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0, 0.0)

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 12:
            x = x[edge:-edge]

        d = x[1:] * np.conj(x[:-1])
        q = np.angle(d)
        n = np.arange(len(q), dtype=float)

        # V2 phase-code detector: deliberately unchanged.
        constant = float(np.median(q))
        r0 = np.angle(np.exp(1j * (q - constant)))
        abs_r0 = np.abs(r0)
        central_limit = float(np.quantile(abs_r0, 0.70))
        central = r0[abs_r0 <= central_limit]
        noise_scale = self._robust_scale(central if len(central) >= 8 else r0)

        jump_threshold = max(5.0 * noise_scale, 0.35)
        jump_mask = abs_r0 > jump_threshold
        jump_excess = np.maximum(abs_r0[jump_mask] - jump_threshold, 0.0)
        if len(jump_excess):
            strongest = np.sort(jump_excess)[-min(8, len(jump_excess)):]
            jump_strength = float(np.sum(strongest) / max(noise_scale, 0.05))
        else:
            jump_strength = 0.0
        phase_coded_score = float(1.0 - np.exp(-jump_strength / 18.0))

        smooth_mask = ~jump_mask
        if np.count_nonzero(smooth_mask) < 8:
            smooth_mask = np.ones_like(jump_mask, dtype=bool)
        ns = n[smooth_mask]
        qs = q[smooth_mask]

        # H0: constant phase increment. H1: constant + linear slope.
        h0 = float(np.mean(qs))
        r_const = np.angle(np.exp(1j * (qs - h0)))
        sse_const = float(np.sum(r_const ** 2))

        a = np.column_stack((np.ones_like(ns), ns))
        coeff, _, _, _ = np.linalg.lstsq(a, qs, rcond=None)
        r_linear = np.angle(np.exp(1j * (qs - a @ coeff)))
        sse_linear = float(np.sum(r_linear ** 2))
        linear_rms = float(np.sqrt(np.mean(r_linear ** 2)))

        count = len(qs)
        # Fraction of residual energy explained by the slope. This is the useful
        # likelihood-ratio-like quantity for our synthetic experiment. Multiply by
        # sample support so weak accidental slopes in noise do not look persuasive.
        explained = max(0.0, sse_const - sse_linear) / max(sse_const, 1e-12)
        support_gain = explained * max(count - 2, 1)
        linear_model_gain = float(support_gain)

        slope_rad_per_sample = float(coeff[1])
        fitted_excursion_rad = abs(slope_rad_per_sample) * max(len(q) - 1, 1)
        excursion_relative_noise = fitted_excursion_rad / max(noise_scale, 1e-4)

        # A true chirp should show both statistically useful linear-model gain and
        # a physically meaningful accumulated phase excursion. The smooth sigmoid
        # forms avoid a brittle single threshold.
        gain_score = support_gain / (support_gain + 4.0)
        excursion_score = excursion_relative_noise / (excursion_relative_noise + 3.0)
        lfm_score = float(np.clip(gain_score * excursion_score, 0.0, 1.0))

        # CW is favoured when the extra slope explains little residual energy.
        cw_score = float(np.clip(1.0 / (1.0 + support_gain / 3.0), 0.0, 1.0))

        suppression = 1.0 - 0.80 * phase_coded_score
        cw_score *= suppression
        lfm_score *= suppression

        scores = {
            "CW": cw_score,
            "LFM": lfm_score,
            "PHASE_CODED": phase_coded_score,
        }
        family = max(scores, key=scores.get)
        if scores[family] < 0.18:
            family = "UNKNOWN"

        slope_hz_per_s = slope_rad_per_sample * self.sample_rate_hz ** 2 / (2.0 * np.pi)
        return PhysicsWaveformEvidenceV3(
            coarse_family=family,
            cw_score=cw_score,
            lfm_score=lfm_score,
            phase_coded_score=phase_coded_score,
            frequency_slope_hz_per_s=float(slope_hz_per_s),
            smooth_residual_rad=linear_rms,
            jump_strength=jump_strength,
            linear_model_gain=linear_model_gain,
        )
