from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandwidthNormalizedClassification:
    family: str
    confidence: float
    family_scores: dict
    occupied_bandwidth_hz: float


def _next_power_of_two(n):
    return 1 << max(0, int(n - 1).bit_length())


class BandwidthNormalizedFFTPCAClassifier:
    """Experimental waveform-family classifier with bandwidth normalization.

    The input pulse may contain a different number of samples from pulse to pulse.
    Each pulse is FFT'd at the next power of two, frequency-centred, its occupied
    bandwidth is estimated, and the spectral shape is resampled onto a fixed
    normalized-frequency grid before PCA. Absolute occupied bandwidth is retained
    as a measured attribute but is not allowed to dominate waveform-family PCA.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, feature_bins=256, n_components=8):
        self.sample_rate_hz = float(sample_rate_hz)
        self.feature_bins = int(feature_bins)
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None
        self.scales_ = None
        self.centroids_ = {}

    def _spectral_feature(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 16:
            return None, 0.0

        nfft = _next_power_of_two(len(x))
        window = np.hanning(len(x))
        spectrum = np.fft.fftshift(np.fft.fft(x * window, n=nfft))
        power = np.abs(spectrum) ** 2
        total = float(np.sum(power))
        if total <= 1e-18:
            return None, 0.0
        power /= total

        bins = np.arange(nfft, dtype=float)
        centroid = float(np.sum(bins * power))
        shift = int(round((nfft - 1) / 2.0 - centroid))
        power = np.roll(power, shift)

        cumulative = np.cumsum(power)
        lo = int(np.searchsorted(cumulative, 0.05))
        hi = int(np.searchsorted(cumulative, 0.95))
        width_bins = max(hi - lo + 1, 3)
        occupied_bandwidth_hz = width_bins * self.sample_rate_hz / nfft

        centre = 0.5 * (nfft - 1)
        half_width = max(width_bins / 2.0, 1.5)
        source_axis = (bins - centre) / half_width
        target_axis = np.linspace(-1.5, 1.5, self.feature_bins)

        log_power = np.log10(power + 1e-12)
        feature = np.interp(
            target_axis,
            source_axis,
            log_power,
            left=log_power[0],
            right=log_power[-1],
        )
        feature -= np.mean(feature)
        norm = np.linalg.norm(feature)
        if norm > 1e-12:
            feature /= norm
        return feature.astype(np.float64), float(occupied_bandwidth_hz)

    def fit(self, samples, labels):
        features = []
        kept_labels = []
        for waveform, label in zip(samples, labels):
            feature, _ = self._spectral_feature(waveform)
            if feature is None:
                continue
            features.append(feature)
            kept_labels.append(label)

        if not features:
            raise ValueError("No valid training waveforms")

        x = np.vstack(features)
        self.mean_ = np.mean(x, axis=0)
        centred = x - self.mean_
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
        count = min(self.n_components, vt.shape[0], vt.shape[1])
        self.components_ = vt[:count]
        self.n_components = count

        denom = max(len(x) - 1, 1)
        variances = (singular_values[:count] ** 2) / denom
        self.scales_ = np.sqrt(np.maximum(variances, 1e-10))

        z = self._project_matrix(x)
        self.centroids_ = {}
        for label in sorted(set(kept_labels)):
            indices = [i for i, item in enumerate(kept_labels) if item == label]
            self.centroids_[label] = np.mean(z[indices], axis=0)
        return self

    def _project_matrix(self, x):
        return ((x - self.mean_) @ self.components_.T) / self.scales_

    def _scores(self, z):
        ranked = []
        for family, centroid in self.centroids_.items():
            distance = float(np.linalg.norm(z - centroid))
            ranked.append((distance, family))
        ranked.sort()

        distances = np.array([item[0] for item in ranked], dtype=float)
        positive = distances[distances > 1e-12]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-9)
        weights = np.exp(-(distances - distances.min()) / scale)
        weights /= weights.sum()
        scores = {family: float(weight) for weight, (_, family) in zip(weights, ranked)}
        return ranked, scores

    def classify(self, samples):
        if self.mean_ is None:
            raise RuntimeError("Classifier must be fit before classify")

        feature, occupied_bandwidth_hz = self._spectral_feature(samples)
        if feature is None:
            return BandwidthNormalizedClassification("UNKNOWN", 0.0, {}, occupied_bandwidth_hz)

        z = self._project_matrix(feature[None, :])[0]
        ranked, scores = self._scores(z)
        best_distance, best_family = ranked[0]
        second_distance = ranked[1][0]
        separation = max(0.0, second_distance - best_distance) / max(second_distance, 1e-9)
        confidence = float(np.clip(0.5 * scores[best_family] + 0.5 * separation, 0.0, 1.0))
        return BandwidthNormalizedClassification(
            family=best_family,
            confidence=confidence,
            family_scores=scores,
            occupied_bandwidth_hz=occupied_bandwidth_hz,
        )
