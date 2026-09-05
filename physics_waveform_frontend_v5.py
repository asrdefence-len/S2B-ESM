from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidenceV5:
    family: str
    scores: dict
    excursion_hz: float
    constant_rms_hz: float
    linear_rms_hz: float
    quadratic_rms_hz: float
    nlfm_improvement: float
    jump_count: int
    median_abs_df_hz: float
    phase_jump_strength: float


class PhysicsWaveformFrontendV5:
    """Trajectory-based broad waveform-family classifier.

    The classifier works on instantaneous-frequency trajectory for CW/LFM/NLFM/
    frequency-coded discrimination, while retaining the V4-style sparse phase-jump
    evidence for PHASE_CODED. It is deliberately broad-family only.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.sample_rate_hz = float(sample_rate_hz)

    @staticmethod
    def _robust_scale(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return 1.0
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return max(1.4826 * mad, 1.0)

    def classify(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 20:
            return PhysicsWaveformEvidenceV5("UNKNOWN", {}, 0.0, np.inf, np.inf, np.inf, 0.0, 0, 0.0, 0.0)

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 16:
            x = x[edge:-edge]

        d = x[1:] * np.conj(x[:-1])
        q_wrapped = np.angle(d)

        # Sparse phase jumps indicate phase coding. Use wrapped residuals around the
        # median phase increment so carrier offset does not matter.
        q_med = float(np.median(q_wrapped))
        phase_resid = np.angle(np.exp(1j * (q_wrapped - q_med)))
        abs_phase_resid = np.abs(phase_resid)
        central = phase_resid[abs_phase_resid <= np.quantile(abs_phase_resid, 0.70)]
        phase_noise = max(1.4826 * float(np.median(np.abs(central - np.median(central)))) if len(central) else 0.0, 1e-4)
        phase_jump_threshold = max(5.0 * phase_noise, 0.35)
        phase_jump_mask = abs_phase_resid > phase_jump_threshold
        phase_excess = np.maximum(abs_phase_resid[phase_jump_mask] - phase_jump_threshold, 0.0)
        if len(phase_excess):
            strongest = np.sort(phase_excess)[-min(12, len(phase_excess)):]
            phase_jump_strength = float(np.sum(strongest) / max(phase_noise, 0.05))
        else:
            phase_jump_strength = 0.0
        phase_score = float(1.0 - np.exp(-phase_jump_strength / 16.0))

        # Remove obvious phase-code discontinuities before analysing instantaneous
        # frequency trajectory. For non-phase-coded pulses this changes little.
        keep = ~phase_jump_mask
        if np.count_nonzero(keep) < 12:
            keep = np.ones_like(phase_jump_mask, dtype=bool)

        q = np.unwrap(q_wrapped)
        f = self.sample_rate_hz * q / (2.0 * np.pi)
        fk = f[keep]
        nk = np.arange(len(f), dtype=float)[keep]
        if len(fk) < 10:
            return PhysicsWaveformEvidenceV5("UNKNOWN", {}, 0.0, np.inf, np.inf, np.inf, 0.0, 0, 0.0, phase_jump_strength)

        n_norm = np.linspace(-1.0, 1.0, len(f))
        nn = n_norm[keep]

        def poly_rms(degree):
            coeff = np.polyfit(nn, fk, degree)
            pred = np.polyval(coeff, nn)
            resid = fk - pred
            return float(np.sqrt(np.mean(resid ** 2)))

        constant_rms = poly_rms(0)
        linear_rms = poly_rms(1)
        quadratic_rms = poly_rms(2)

        excursion = float(np.percentile(fk, 95) - np.percentile(fk, 5))
        nlfm_improvement = max(0.0, linear_rms - quadratic_rms) / max(linear_rms, 1.0)

        df = np.diff(f)
        if len(df):
            med_df = float(np.median(df))
            mad_df = float(np.median(np.abs(df - med_df)))
            df_scale = max(1.4826 * mad_df, 1.0)
            jump_threshold_hz = max(6.0 * df_scale, 50_000.0)
            jump_mask = np.abs(df - med_df) > jump_threshold_hz
            jump_count = int(np.count_nonzero(jump_mask))
            median_abs_df = float(np.median(np.abs(df)))
        else:
            jump_count = 0
            median_abs_df = 0.0

        # Dimensionless trajectory evidence. Scales are intentionally broad and
        # physics-driven starting points, not calibrated probabilities.
        excursion_scale = max(excursion, 1.0)
        flatness = 1.0 / (1.0 + constant_rms / max(0.03 * excursion_scale, 20_000.0))
        cw_score = float(np.clip(flatness, 0.0, 1.0))

        linear_quality = 1.0 / (1.0 + linear_rms / max(0.04 * excursion_scale, 20_000.0))
        quadratic_quality = 1.0 / (1.0 + quadratic_rms / max(0.04 * excursion_scale, 20_000.0))
        excursion_presence = excursion / (excursion + 300_000.0)

        lfm_score = float(np.clip(excursion_presence * linear_quality * (1.0 - 0.85 * nlfm_improvement), 0.0, 1.0))
        nlfm_score = float(np.clip(excursion_presence * quadratic_quality * nlfm_improvement, 0.0, 1.0))

        # Frequency coding: long plateaus imply tiny median |df|, while sparse large
        # jumps account for the total excursion. Penalise many small changes (chirp).
        plateau_score = 1.0 / (1.0 + median_abs_df / max(0.01 * excursion_scale, 5_000.0))
        sparse_jump_score = float(np.clip(jump_count / 3.0, 0.0, 1.0)) * float(np.clip(12.0 / max(jump_count, 1), 0.0, 1.0))
        freq_code_score = float(np.clip(excursion_presence * plateau_score * sparse_jump_score, 0.0, 1.0))

        # Strong phase-code evidence should dominate the trajectory families.
        smooth_suppression = 1.0 - 0.85 * phase_score
        cw_score *= smooth_suppression
        lfm_score *= smooth_suppression
        nlfm_score *= smooth_suppression
        freq_code_score *= smooth_suppression

        scores = {
            "CW": cw_score,
            "LFM": lfm_score,
            "NLFM": nlfm_score,
            "FREQUENCY_CODED": freq_code_score,
            "PHASE_CODED": phase_score,
        }

        family = max(scores, key=scores.get)
        ranked = sorted(scores.values(), reverse=True)
        best = ranked[0]
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        if best < 0.20 or (best < 0.35 and margin < 0.03):
            family = "UNKNOWN"

        return PhysicsWaveformEvidenceV5(
            family=family,
            scores=scores,
            excursion_hz=excursion,
            constant_rms_hz=constant_rms,
            linear_rms_hz=linear_rms,
            quadratic_rms_hz=quadratic_rms,
            nlfm_improvement=nlfm_improvement,
            jump_count=jump_count,
            median_abs_df_hz=median_abs_df,
            phase_jump_strength=phase_jump_strength,
        )
