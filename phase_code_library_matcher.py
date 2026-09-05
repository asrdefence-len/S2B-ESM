from dataclasses import dataclass

import numpy as np

from cyclostationary_waveform_diagnostic import cyclic_features


@dataclass(frozen=True)
class PhaseCodeLibraryMatch:
    accepted: bool
    library_id: str
    distance_ratio: float
    distance: float
    threshold: float


class PhaseCodeLibraryMatcher:
    """Strict matcher for known phase-coded waveform entries.

    Unlike the generic FFT/PCA matcher, this feature space explicitly uses phase
    geometry: detrended residual-phase histograms, differential-phase histograms,
    transition density and cyclic x^2/x^4 features. It is intended to distinguish
    known Barker/generic-biphase/Frank-like entries from arbitrary polyphase and
    hybrid FM+phase waveforms.
    """

    def __init__(self, pca_dims=12):
        self.pca_dims = int(pca_dims)
        self.mean_ = None
        self.components_ = None
        self.scales_ = None
        self.vectors_ = {}
        self.thresholds_ = {}

    @staticmethod
    def _circular_hist(values, bins):
        values = np.angle(np.exp(1j * np.asarray(values, dtype=float)))
        hist, _ = np.histogram(values, bins=bins, range=(-np.pi, np.pi))
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _feature(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 24:
            return None

        # Remove edge samples where pulse timing uncertainty is most damaging.
        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 20:
            x = x[edge:-edge]

        phase = np.unwrap(np.angle(x))
        n = np.linspace(-1.0, 1.0, len(phase))

        # Remove any smooth carrier/FM trend before examining code phase states.
        degree = min(3, max(1, len(phase) // 40))
        coeff = np.polyfit(n, phase, degree)
        residual = np.angle(np.exp(1j * (phase - np.polyval(coeff, n))))

        dphi = np.angle(x[1:] * np.conj(x[:-1]))
        med = float(np.median(dphi))
        dres = np.angle(np.exp(1j * (dphi - med)))

        phase_hist = self._circular_hist(residual, 16)
        diff_hist = self._circular_hist(dres, 16)

        # Transition statistics: known codes have characteristic jump density and
        # jump-angle structure. Threshold is deliberately noise-tolerant.
        mag = np.abs(dres)
        central = dres[mag <= np.quantile(mag, 0.70)]
        if len(central):
            mad = float(np.median(np.abs(central - np.median(central))))
        else:
            mad = 0.0
        noise = max(1.4826 * mad, 1e-4)
        jump_threshold = max(5.0 * noise, 0.35)
        jumps = dres[mag > jump_threshold]
        transition_fraction = float(len(jumps) / max(len(dres), 1))
        jump_hist = self._circular_hist(jumps if len(jumps) else np.array([0.0]), 12)

        # Coarse phase-state occupancy after detrending. This is not an exact chip
        # decoder; it simply records whether the residual phase occupies a few
        # repeatable states or a more arbitrary distribution.
        coarse = np.round((residual + np.pi) / (np.pi / 8.0)).astype(int) % 16
        _, counts = np.unique(coarse, return_counts=True)
        p = counts.astype(float) / counts.sum()
        state_entropy = float(-np.sum(p * np.log(p + 1e-15)) / np.log(16.0))
        state_count = float(np.count_nonzero(counts >= max(2, int(0.02 * len(residual))))) / 16.0

        cf = cyclic_features(x)
        aux = np.array([
            transition_fraction,
            state_entropy,
            state_count,
            cf["x2_peak"], cf["x2_top4"], cf["x2_entropy"],
            cf["x4_peak"], cf["x4_top4"], cf["x4_entropy"],
        ], dtype=float)

        return np.concatenate((phase_hist, diff_hist, jump_hist, aux))

    def fit(self, samples, library_ids):
        features = []
        ids = []
        for waveform, library_id in zip(samples, library_ids):
            feature = self._feature(waveform)
            if feature is None:
                continue
            features.append(feature)
            ids.append(str(library_id))
        if not features:
            raise ValueError("No valid phase-code library examples")

        x = np.vstack(features)
        self.mean_ = np.mean(x, axis=0)
        centred = x - self.mean_
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
        count = min(self.pca_dims, vt.shape[0], vt.shape[1])
        self.components_ = vt[:count]
        denom = max(len(x) - 1, 1)
        variances = (singular_values[:count] ** 2) / denom
        self.scales_ = np.sqrt(np.maximum(variances, 1e-10))
        z = self._project(x)

        for library_id in sorted(set(ids)):
            idx = [i for i, value in enumerate(ids) if value == library_id]
            vectors = z[idx]
            self.vectors_[library_id] = vectors
            nn = []
            for i in range(len(vectors)):
                if len(vectors) < 2:
                    continue
                d = np.linalg.norm(vectors - vectors[i], axis=1)
                d[i] = np.inf
                nn.append(float(np.min(d)))
            self.thresholds_[library_id] = (
                1.20 * float(np.quantile(nn, 0.98)) if nn else np.inf
            )
        return self

    def _project(self, x):
        return ((x - self.mean_) @ self.components_.T) / self.scales_

    def match(self, samples):
        if self.mean_ is None:
            raise RuntimeError("Phase-code matcher must be fit before match")
        feature = self._feature(samples)
        if feature is None:
            return PhaseCodeLibraryMatch(False, "NONE", np.inf, np.inf, 0.0)
        z = self._project(feature[None, :])[0]
        candidates = []
        for library_id, vectors in self.vectors_.items():
            distance = float(np.min(np.linalg.norm(vectors - z, axis=1)))
            threshold = max(float(self.thresholds_[library_id]), 1e-9)
            candidates.append((distance / threshold, distance, threshold, library_id))
        candidates.sort(key=lambda item: item[0])
        ratio, distance, threshold, library_id = candidates[0]
        return PhaseCodeLibraryMatch(
            accepted=bool(ratio <= 1.0),
            library_id=library_id,
            distance_ratio=float(ratio),
            distance=float(distance),
            threshold=float(threshold),
        )
