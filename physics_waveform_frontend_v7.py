"""V7 physics waveform frontend with vectorized local-frequency regression.

Classification/scoring is intentionally the same as V6.  The only substantive
change is _local_frequency(): V6 solved a least-squares problem independently at
every sliding-window position; V7 uses the closed-form slope weights for equally
spaced samples and np.correlate to evaluate every window at once.
"""

import numpy as np

from physics_waveform_frontend_v6 import PhysicsWaveformEvidenceV6, PhysicsWaveformFrontendV6


class PhysicsWaveformFrontendV7(PhysicsWaveformFrontendV6):
    """Drop-in V6-compatible frontend with a vectorized local-frequency estimator."""

    def _local_frequency(self, phase):
        phase = np.asarray(phase, dtype=float)
        window = max(9, int(round(len(phase) * 0.08)))
        if window % 2 == 0:
            window += 1
        if len(phase) < window:
            return np.asarray([], dtype=float)

        # For y = a + b*x on an equally spaced window, after centering x the
        # least-squares slope is simply sum(xc*y) / sum(xc^2).  The weights are
        # fixed for every window, so all sliding slopes can be evaluated by one
        # vectorized correlation. This is algebraically equivalent to V6's LSQ.
        x = np.arange(window, dtype=float)
        xc = x - np.mean(x)
        denom = float(np.dot(xc, xc))
        weights = xc / denom
        slopes_rad_per_sample = np.correlate(phase, weights, mode="valid")
        return slopes_rad_per_sample * self.sample_rate_hz / (2.0 * np.pi)

    def classify(self, samples):
        # Keep a local copy of V6 classify logic so this class remains explicit
        # about the algorithm under regression. Only _local_frequency differs.
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
            plateau_threshold = max(0.003 * max(excursion, 1.0), 5_000.0)
            plateau_fraction = float(np.mean(np.abs(df) < plateau_threshold))
        else:
            excursion = 0.0
            plateau_fraction = 0.0

        cw_score = float(1.0 / (1.0 + rms1 / 0.50))
        best_fm_rms = min(rms2, rms3)
        fm_improvement = max(0.0, rms1 - best_fm_rms) / max(rms1, 1e-6)
        fm_fit_quality = 1.0 / (1.0 + best_fm_rms / 0.75)
        fm_score = float(np.clip(fm_improvement * fm_fit_quality, 0.0, 1.0))
        freq_code_score = float(np.clip(plateau_fraction, 0.0, 1.0))

        smooth_suppression = 1.0 - 0.85 * phase_score
        cw_score *= smooth_suppression
        fm_score *= smooth_suppression
        freq_code_score *= smooth_suppression
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
