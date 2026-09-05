from dataclasses import dataclass

import numpy as np

from bandwidth_normalized_fft_pca_classifier import BandwidthNormalizedFFTPCAClassifier
from cyclostationary_waveform_diagnostic import cyclic_features


@dataclass(frozen=True)
class WaveformLibraryMatch:
    accepted: bool
    library_id: str
    broad_family: str
    distance_ratio: float
    distance: float
    threshold: float


class WaveformLibraryMatcher:
    """Feature-library matcher for strict known-waveform acceptance.

    The broad physics classifier answers CW/FM/PHASE_CODED. This matcher answers a
    different question: is the observed pulse sufficiently close to one of the
    waveform types in the current library? If not, the operational result should
    be UNKNOWN.

    Features combine bandwidth-normalized FFT shape, lightweight cyclic features,
    and a few amplitude statistics. Distances are evaluated in a whitened PCA space.
    Each library entry gets its own empirical acceptance radius from leave-one-out
    nearest-neighbour distances among its training examples.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, spectral_bins=128, pca_dims=16):
        self.sample_rate_hz = float(sample_rate_hz)
        self.spectral = BandwidthNormalizedFFTPCAClassifier(
            sample_rate_hz=sample_rate_hz,
            feature_bins=spectral_bins,
            n_components=pca_dims,
        )
        self.pca_dims = int(pca_dims)
        self.mean_ = None
        self.components_ = None
        self.scales_ = None
        self.library_vectors_ = {}
        self.thresholds_ = {}
        self.family_by_id_ = {}

    def _feature(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        spectral, occupied_bw = self.spectral._spectral_feature(x)
        if spectral is None:
            return None

        cf = cyclic_features(x)
        mag = np.abs(x)
        mag_mean = float(np.mean(mag))
        mag_std = float(np.std(mag) / max(mag_mean, 1e-9))
        mag_p2p = float((np.percentile(mag, 95) - np.percentile(mag, 5)) / max(mag_mean, 1e-9))

        aux = np.array([
            cf["x2_peak"], cf["x2_top4"], cf["x2_entropy"],
            cf["x4_peak"], cf["x4_top4"], cf["x4_entropy"],
            cf["env_peak"], cf["env_top4"], cf["env_entropy"],
            np.log10(max(occupied_bw, 1.0)), mag_std, mag_p2p,
        ], dtype=float)
        return np.concatenate((spectral, aux))

    def fit(self, samples, library_ids, broad_families):
        features = []
        ids = []
        families = []
        for waveform, library_id, broad_family in zip(samples, library_ids, broad_families):
            feature = self._feature(waveform)
            if feature is None:
                continue
            features.append(feature)
            ids.append(str(library_id))
            families.append(str(broad_family))

        if not features:
            raise ValueError("No valid waveform-library training examples")

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

        self.library_vectors_.clear()
        self.thresholds_.clear()
        self.family_by_id_.clear()

        for library_id in sorted(set(ids)):
            idx = [i for i, item in enumerate(ids) if item == library_id]
            vectors = z[idx]
            self.library_vectors_[library_id] = vectors
            family_values = [families[i] for i in idx]
            self.family_by_id_[library_id] = max(set(family_values), key=family_values.count)

            # Leave-one-out same-library nearest-neighbour distances define the
            # empirical known-waveform envelope. Use a high quantile plus margin.
            nn = []
            for i in range(len(vectors)):
                if len(vectors) < 2:
                    continue
                d = np.linalg.norm(vectors - vectors[i], axis=1)
                d[i] = np.inf
                nn.append(float(np.min(d)))
            if nn:
                self.thresholds_[library_id] = 1.35 * float(np.quantile(nn, 0.98))
            else:
                self.thresholds_[library_id] = np.inf
        return self

    def _project(self, x):
        return ((x - self.mean_) @ self.components_.T) / self.scales_

    def match(self, samples, broad_family=None):
        if self.mean_ is None:
            raise RuntimeError("Waveform library must be fit before match")
        feature = self._feature(samples)
        if feature is None:
            return WaveformLibraryMatch(False, "NONE", "UNKNOWN", np.inf, np.inf, 0.0)
        z = self._project(feature[None, :])[0]

        candidates = []
        for library_id, vectors in self.library_vectors_.items():
            family = self.family_by_id_[library_id]
            if broad_family is not None and family != broad_family:
                continue
            nearest = float(np.min(np.linalg.norm(vectors - z, axis=1)))
            threshold = max(float(self.thresholds_[library_id]), 1e-9)
            candidates.append((nearest / threshold, nearest, threshold, library_id, family))

        if not candidates:
            return WaveformLibraryMatch(False, "NONE", "UNKNOWN", np.inf, np.inf, 0.0)

        candidates.sort(key=lambda item: item[0])
        ratio, distance, threshold, library_id, family = candidates[0]
        return WaveformLibraryMatch(
            accepted=bool(ratio <= 1.0),
            library_id=library_id,
            broad_family=family,
            distance_ratio=float(ratio),
            distance=float(distance),
            threshold=float(threshold),
        )
