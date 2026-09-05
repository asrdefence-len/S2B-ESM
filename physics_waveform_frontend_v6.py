from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsWaveformEvidenceV6:
    family: str
    scores: dict
    phase_rms_linear: float
    phase_rms_quadratic: float
    phase_rms_cubic: float
    phase_jump_strength: float
    local_frequency_excursion_hz: float
    local_frequency_plateau_fraction: float


class PhysicsWaveformFrontendV6:
    """Broad waveform-family classifier for the operational ESM path.

    Current broad classes:
      CW
      FM                 (LFM and NLFM intentionally combined)
      FREQUENCY_CODED
      PHASE_CODED
      UNKNOWN

    Exact waveform fingerprinting is deliberately deferred. The classifier keeps
    phase-fit and local-frequency observables so later S2B logic can still detect
    changes within a broad family.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.sample_rate_hz = float(sample_rate_hz)

    @staticmethod
    def _wrapped_phase_jump_score(x):
        d = x[1:] * np.conj(x[:-1])
        q = np.angle(d)
        med = float(np.median(q))
        resid = np.angle(np.exp(1j * (q - med)))
        a = np.abs(resid)
        central = resid[a <= np.quantile(a, 0.70)]
        if len(central):
            mad = float(np.median(np.abs(central - np.median(central))))
        else:
            mad = 0.0
        noise = max(1.4826 * mad, 1e-4)
        threshold = max(5.0 * noise, 0.35)
        excess = np.maximum(a[a > threshold] - threshold, 0.0)
        if not len(excess):
            return 0.0
        strongest = np.sort(excess)[-min(12, len(excess)):]
        strength = float(np.sum(strongest) / max(noise, 0.05))
        return float(1.0 - np.exp(-strength / 16.0))

    @staticmethod
    def _phase_fit_rms(phase, degree):
        n = np.linspace(-1.0, 1.0, len(phase))
        coeff = np.polyfit(n, phase, degree)
        pred = np.polyval(coeff, n)
        resid = phase - pred
        return float(np.sqrt(np.mean(resid ** 2)))

    def _local_frequency(self, phase):
        window = max(9, int(round(len(phase) * 0.08)))
        if window % 2 == 0:
            window += 1
        half = window // 2
        slopes = []
        for i in range(half, len(phase) - half):
            idx = np.arange(i - half, i + half + 1, dtype=float)
            y = phase[i - half:i + half + 1]
            a = np.column_stack((np.ones_like(idx), idx))
            coeff, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
            slopes.append(coeff[1] * self.sample_rate_hz / (2.0 * np.pi))
        return np.asarray(slopes, dtype=float)

    def classify(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 24:
            return PhysicsWaveformEvidenceV6("UNKNOWN", {}, np.inf, np.inf, np.inf, 0.0, 0.0, 0.0)

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 20:
            x = x[edge:-edge]

        phase_score = self._wrapped_phase_jump_score(x)
        phase = np.unwrap(np.angle(x))

        rms1 = self._phase_fit_rms(phase, 1)
        rms2 = self._phase_fit_rms(phase, 2)
        rms3 = self._phase_fit_rms(phase, 3)

        local_f = self._local_frequency(phase)
        if len(local_f) >= 4:
            excursion = float(np.percentile(local_f, 95) - np.percentile(local_f, 5))
            df = np.diff(local_f)
            # Plateau fraction: samples whose local-frequency slope is very small
            # compared with the overall excursion. Frequency-coded waveforms spend
            # much of the pulse on plateaus; continuous FM does not.
            plateau_threshold = max(0.003 * max(excursion, 1.0), 5_000.0)
            plateau_fraction = float(np.mean(np.abs(df) < plateau_threshold))
        else:
            excursion = 0.0
            plateau_fraction = 0.0

        # Phase-polynomial model evidence. CW requires a good linear phase fit.
        cw_score = float(1.0 / (1.0 + rms1 / 0.50))

        # FM includes both LFM and NLFM: either quadratic or cubic phase should fit
        # substantially better than a linear phase model.
        best_fm_rms = min(rms2, rms3)
        fm_improvement = max(0.0, rms1 - best_fm_rms) / max(rms1, 1e-6)
        fm_fit_quality = 1.0 / (1.0 + best_fm_rms / 0.75)
        fm_score = float(np.clip(fm_improvement * fm_fit_quality, 0.0, 1.0))

        # Frequency coding can also improve under polynomial fits, so use local
        # frequency plateau structure to separate it from continuous FM.
        freq_code_score = float(np.clip(plateau_fraction, 0.0, 1.0))

        # Strong phase discontinuities should dominate the smooth-family scores.
        smooth_suppression = 1.0 - 0.85 * phase_score
        cw_score *= smooth_suppression
        fm_score *= smooth_suppression
        freq_code_score *= smooth_suppression

        # Penalise FM when the local-frequency trajectory is predominantly flat.
        fm_score *= (1.0 - 0.75 * plateau_fraction)

        scores = {
            "CW": cw_score,
            "FM": fm_score,
            "FREQUENCY_CODED": freq_code_score,
            "PHASE_CODED": phase_score,
        }

        family = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        if best < 0.20 or (best < 0.38 and margin < 0.04):
            family = "UNKNOWN"

        return PhysicsWaveformEvidenceV6(
            family=family,
            scores=scores,
            phase_rms_linear=rms1,
            phase_rms_quadratic=rms2,
            phase_rms_cubic=rms3,
            phase_jump_strength=phase_score,
            local_frequency_excursion_hz=excursion,
            local_frequency_plateau_fraction=plateau_fraction,
        )
