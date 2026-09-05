from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidence:
    coarse_family: str
    cw_score: float
    lfm_score: float
    phase_coded_score: float
    frequency_slope_hz_per_s: float
    frequency_residual_hz: float


class PhysicsWaveformFrontend:
    """Coarse intra-pulse classifier based on instantaneous-frequency structure.

    This is deliberately a front end, not the final waveform classifier. It asks
    whether the pulse is best explained by constant instantaneous frequency (CW),
    a linear instantaneous-frequency ramp (LFM), or neither (phase-coded/other).
    Carrier offset is absorbed by the fitted intercept.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.sample_rate_hz = float(sample_rate_hz)

    def classify(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 16:
            return PhysicsWaveformEvidence("UNKNOWN", 0.0, 0.0, 0.0, 0.0, float("inf"))

        # Trim edges because pulse-window truncation/rise/fall can dominate phase
        # differences there. Differential phase removes arbitrary carrier phase.
        edge = max(2, int(round(0.05 * len(x))))
        if len(x) - 2 * edge >= 12:
            x = x[edge:-edge]

        d = x[1:] * np.conj(x[:-1])
        phase_step = np.angle(d)
        inst_freq = phase_step * self.sample_rate_hz / (2.0 * np.pi)

        # Robustly suppress the largest phase discontinuities before testing CW/LFM.
        # Those discontinuities are themselves evidence for phase coding.
        median = float(np.median(inst_freq))
        mad = float(np.median(np.abs(inst_freq - median)))
        robust_sigma = max(1.4826 * mad, self.sample_rate_hz * 1e-5)
        jump_mask = np.abs(inst_freq - median) > 5.0 * robust_sigma
        clean = inst_freq[~jump_mask]
        clean_indices = np.nonzero(~jump_mask)[0].astype(float)

        jump_fraction = float(np.mean(jump_mask))
        phase_coded_score = float(np.clip(jump_fraction / 0.08, 0.0, 1.0))

        if len(clean) < 8:
            return PhysicsWaveformEvidence(
                "PHASE_CODED", 0.0, 0.0, max(phase_coded_score, 0.8), 0.0, float("inf")
            )

        t = clean_indices / self.sample_rate_hz
        a = np.column_stack((np.ones_like(t), t))
        coeff, _, _, _ = np.linalg.lstsq(a, clean, rcond=None)
        fitted = a @ coeff
        residual = clean - fitted
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))
        slope = float(coeff[1])

        # Constant-frequency model residual.
        cw_residual = clean - np.mean(clean)
        cw_rms = float(np.sqrt(np.mean(cw_residual ** 2)))

        # Compare signal frequency excursion to residual noise. LFM needs a real
        # fitted excursion; CW needs little benefit from adding the slope term.
        duration = max((len(x) - 1) / self.sample_rate_hz, 1.0 / self.sample_rate_hz)
        fitted_excursion = abs(slope) * duration
        lfm_strength = fitted_excursion / max(residual_rms, 1.0)
        fit_improvement = max(0.0, cw_rms - residual_rms) / max(cw_rms, 1.0)

        lfm_score = float(np.clip(0.55 * (lfm_strength / (lfm_strength + 3.0)) + 0.45 * fit_improvement, 0.0, 1.0))
        cw_score = float(np.clip((1.0 - fit_improvement) * (1.0 / (1.0 + cw_rms / 2.0e5)), 0.0, 1.0))

        # Explicit phase discontinuities should override an apparently good smooth
        # fit. This keeps Barker/polyphase pulses out of CW/LFM where possible.
        cw_score *= (1.0 - 0.85 * phase_coded_score)
        lfm_score *= (1.0 - 0.85 * phase_coded_score)

        if phase_coded_score >= max(cw_score, lfm_score) and phase_coded_score >= 0.30:
            family = "PHASE_CODED"
        elif lfm_score > cw_score and lfm_score >= 0.35:
            family = "LFM"
        elif cw_score >= 0.35:
            family = "CW"
        else:
            family = "UNKNOWN"

        return PhysicsWaveformEvidence(
            coarse_family=family,
            cw_score=cw_score,
            lfm_score=lfm_score,
            phase_coded_score=phase_coded_score,
            frequency_slope_hz_per_s=slope,
            frequency_residual_hz=residual_rms,
        )
