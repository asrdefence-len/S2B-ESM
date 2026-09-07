"""V8 physics waveform frontend with cached polynomial projection matrices.

V8 preserves V7 scoring and local-frequency processing.  It replaces repeated
np.polyfit/np.polyval calls with cached least-squares projection matrices keyed by
(phase length, degree). Pulse lengths repeat heavily in the streaming ESM, so the
small projection matrices are constructed once and then reused.
"""

import numpy as np

from physics_waveform_frontend_v7 import PhysicsWaveformFrontendV7


class PhysicsWaveformFrontendV8(PhysicsWaveformFrontendV7):
    """V7-compatible frontend with cached degree-1/2/3 phase-fit projections."""

    def __init__(self, sample_rate_hz=40_000_000.0):
        super().__init__(sample_rate_hz)
        self._phase_projection_cache = {}

    def _projection(self, length, degree):
        key = (int(length), int(degree))
        projection = self._phase_projection_cache.get(key)
        if projection is None:
            n = np.linspace(-1.0, 1.0, length)
            # np.polyfit is least squares on the Vandermonde design matrix.  The
            # hat/projection matrix H = A pinv(A) gives identical fitted samples
            # (within floating precision) without resolving the fit every pulse.
            design = np.vander(n, degree + 1)
            projection = design @ np.linalg.pinv(design)
            self._phase_projection_cache[key] = projection
        return projection

    def _phase_fit_rms(self, phase, degree):
        phase = np.asarray(phase, dtype=float)
        if not len(phase):
            return float("inf")
        pred = self._projection(len(phase), degree) @ phase
        resid = phase - pred
        return float(np.sqrt(np.mean(resid ** 2)))
