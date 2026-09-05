from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidenceV2:
    coarse_family: str
    cw_score: float
    lfm_score: float
    phase_coded_score: float
    frequency_slope_hz_per_s: float
    smooth_residual_rad: float
    jump_strength: float


class PhysicsWaveformFrontendV2:
    """Model-based coarse waveform classifier.

    Competing explanations are evaluated from differential phase:
      CW          : phase increment is approximately constant
      LFM         : phase increment is approximately linear with sample index
      PHASE_CODED : mostly smooth phase increment plus sparse large discontinuities

    Scores are relative to the pulse's own robust phase-noise scale rather than an
    absolute frequency-residual threshold. Phase-code evidence uses jump strength,
    not jump count divided by pulse length, reducing pulse-width dependence.
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
            return PhysicsWaveformEvidenceV2("UNKNOWN", 0.0, 0.0, 0.0, 0.0, float("inf"), 0.0)

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 12:
            x = x[edge:-edge]

        # Differential phase removes arbitrary absolute phase. Unwrap the phase
        # increment sequence only locally through robust model fitting; genuine
        # code transitions remain large residual events.
        d = x[1:] * np.conj(x[:-1])
        q = np.angle(d)
        n = np.arange(len(q), dtype=float)

        # First robust constant model. Estimate the smooth-sample noise scale from
        # the central part of the residual distribution rather than from all jumps.
        constant = float(np.median(q))
        r0 = np.angle(np.exp(1j * (q - constant)))
        abs_r0 = np.abs(r0)
        central_limit = float(np.quantile(abs_r0, 0.70))
        central = r0[abs_r0 <= central_limit]
        noise_scale = self._robust_scale(central if len(central) >= 8 else r0)

        # Large residual phase events are candidate code transitions. Use their
        # excess magnitude, normalized by noise scale, rather than their fraction
        # of total pulse samples.
        jump_threshold = max(5.0 * noise_scale, 0.35)
        jump_mask = abs_r0 > jump_threshold
        jump_excess = np.maximum(abs_r0[jump_mask] - jump_threshold, 0.0)
        if len(jump_excess):
            strongest = np.sort(jump_excess)[-min(8, len(jump_excess)):]
            jump_strength = float(np.sum(strongest) / max(noise_scale, 0.05))
        else:
            jump_strength = 0.0
        phase_coded_score = float(1.0 - np.exp(-jump_strength / 18.0))

        # Fit smooth models after removing candidate discontinuities. A phase code
        # should leave a nearly constant background once its sparse jumps are removed.
        smooth_mask = ~jump_mask
        if np.count_nonzero(smooth_mask) < 8:
            smooth_mask = np.ones_like(jump_mask, dtype=bool)

        ns = n[smooth_mask]
        qs = q[smooth_mask]
        a = np.column_stack((np.ones_like(ns), ns))
        coeff, _, _, _ = np.linalg.lstsq(a, qs, rcond=None)
        linear_fit = a @ coeff
        linear_residual = np.angle(np.exp(1j * (qs - linear_fit)))
        linear_rms = float(np.sqrt(np.mean(linear_residual ** 2)))

        constant_fit = float(np.mean(qs))
        constant_residual = np.angle(np.exp(1j * (qs - constant_fit)))
        constant_rms = float(np.sqrt(np.mean(constant_residual ** 2)))

        improvement = max(0.0, constant_rms - linear_rms) / max(constant_rms, noise_scale)
        slope_rad_per_sample = float(coeff[1])
        fitted_excursion_rad = abs(slope_rad_per_sample) * max(len(q) - 1, 1)
        excursion_snr = fitted_excursion_rad / max(noise_scale, 1e-4)

        # CW is the null model: smooth phase increment with little justified linear
        # trend. LFM requires both substantial excursion and improved linear fit.
        smoothness = 1.0 / (1.0 + linear_rms / max(noise_scale, 1e-4))
        lfm_score = float(
            np.clip(
                (excursion_snr / (excursion_snr + 8.0))
                * (0.35 + 0.65 * improvement)
                * smoothness,
                0.0,
                1.0,
            )
        )
        cw_score = float(
            np.clip(
                (1.0 - improvement)
                * (1.0 / (1.0 + excursion_snr / 8.0))
                * smoothness,
                0.0,
                1.0,
            )
        )

        # Strong sparse jumps should dominate smooth-model decisions.
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
        return PhysicsWaveformEvidenceV2(
            coarse_family=family,
            cw_score=cw_score,
            lfm_score=lfm_score,
            phase_coded_score=phase_coded_score,
            frequency_slope_hz_per_s=float(slope_hz_per_s),
            smooth_residual_rad=linear_rms,
            jump_strength=jump_strength,
        )
