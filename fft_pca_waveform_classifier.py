from dataclasses import dataclass

import numpy as np

from waveform_library import barker13, biphase, cw, lfm, polyphase


@dataclass(frozen=True)
class FFTPCAClassification:
    family: str
    subtype: str
    confidence: float
    distance: float
    runner_up_family: str
    runner_up_subtype: str
    runner_up_distance: float
    family_scores: dict


def _next_power_of_two(n):
    return 1 << max(0, int(n - 1).bit_length())


class FFTPCAWaveformClassifier:
    """Experimental FFT-magnitude/PCA waveform classifier.

    The representation is intentionally insensitive to common carrier phase and
    approximately insensitive to pulse timing translation because it uses FFT
    magnitude rather than complex FFT phase. Frequency translation is removed by
    centring the spectrum on its power centroid before PCA.

    PCA is implemented with NumPy SVD so this experiment adds no sklearn
    dependency. Classification is nearest class centroid in whitened PCA space.
    The returned family_scores retain soft evidence across all waveform families.
    They sum to one, but are engineering evidence weights rather than calibrated
    posterior probabilities.
    """

    def __init__(self, sample_rate_hz=40_000_000.0, n_components=8):
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None
        self.scales_ = None
        self.centroids_ = {}
        self.subtypes_ = {}

    @staticmethod
    def _spectral_feature(samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 8:
            return None

        nfft = _next_power_of_two(len(x))
        window = np.hanning(len(x))
        spectrum = np.fft.fftshift(np.fft.fft(x * window, n=nfft))
        power = np.abs(spectrum) ** 2
        total = float(np.sum(power))
        if total <= 1e-18:
            return None
        power = power / total

        bins = np.arange(nfft, dtype=float)
        centroid = float(np.sum(bins * power))
        shift = int(round((nfft - 1) / 2.0 - centroid))
        power = np.roll(power, shift)

        feature = np.log10(power + 1e-12)
        feature -= np.mean(feature)
        norm = np.linalg.norm(feature)
        if norm > 1e-12:
            feature /= norm
        return feature.astype(np.float64)

    def fit(self, samples, labels, subtypes=None):
        features = []
        kept_labels = []
        kept_subtypes = []

        if subtypes is None:
            subtypes = labels

        for waveform, label, subtype in zip(samples, labels, subtypes):
            feature = self._spectral_feature(waveform)
            if feature is None:
                continue
            features.append(feature)
            kept_labels.append(label)
            kept_subtypes.append(subtype)

        if not features:
            raise ValueError("No valid training waveforms")

        widths = {len(feature) for feature in features}
        if len(widths) != 1:
            raise ValueError("All training pulses must produce the same FFT size")

        x = np.vstack(features)
        self.mean_ = np.mean(x, axis=0)
        centred = x - self.mean_
        _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
        count = min(self.n_components, vt.shape[0], vt.shape[1])
        self.components_ = vt[:count]

        denom = max(len(x) - 1, 1)
        variances = (singular_values[:count] ** 2) / denom
        self.scales_ = np.sqrt(np.maximum(variances, 1e-10))

        z = self._project_matrix(x)
        self.centroids_.clear()
        self.subtypes_.clear()
        for label in sorted(set(kept_labels)):
            indices = [i for i, item in enumerate(kept_labels) if item == label]
            self.centroids_[label] = np.mean(z[indices], axis=0)
            subtype_counts = {}
            for i in indices:
                subtype = kept_subtypes[i]
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            self.subtypes_[label] = max(subtype_counts, key=subtype_counts.get)
        return self

    def _project_matrix(self, x):
        return ((x - self.mean_) @ self.components_.T) / self.scales_

    def _project(self, feature):
        return self._project_matrix(feature[None, :])[0]

    @staticmethod
    def _soft_scores(ranked):
        """Convert centroid distances to relative evidence weights.

        Use distance relative to the nearest centroid so the exponentials remain
        numerically stable. The scale is the median non-zero distance, which
        adapts to the local geometry without claiming probability calibration.
        """
        distances = np.array([distance for distance, _ in ranked], dtype=float)
        positive = distances[distances > 1e-12]
        scale = float(np.median(positive)) if len(positive) else 1.0
        scale = max(scale, 1e-9)
        relative = distances - np.min(distances)
        weights = np.exp(-relative / scale)
        weights /= max(float(np.sum(weights)), 1e-12)
        return {
            family: float(weight)
            for weight, (_, family) in zip(weights, ranked)
        }

    def classify(self, samples):
        if self.mean_ is None:
            raise RuntimeError("Classifier must be fit before classify")

        feature = self._spectral_feature(samples)
        if feature is None or len(feature) != len(self.mean_):
            return FFTPCAClassification(
                "UNKNOWN", "UNKNOWN", 0.0, float("inf"),
                "UNKNOWN", "UNKNOWN", float("inf"), {},
            )

        z = self._project(feature)
        ranked = []
        for family, centroid in self.centroids_.items():
            distance = float(np.linalg.norm(z - centroid))
            ranked.append((distance, family))
        ranked.sort()

        best_distance, best_family = ranked[0]
        second_distance, second_family = ranked[1]
        family_scores = self._soft_scores(ranked)

        closeness = 1.0 / (1.0 + best_distance)
        separation = max(0.0, second_distance - best_distance) / max(second_distance, 1e-9)
        confidence = float(np.clip(0.6 * closeness + 0.4 * separation, 0.0, 1.0))

        return FFTPCAClassification(
            family=best_family,
            subtype=self.subtypes_[best_family],
            confidence=confidence,
            distance=best_distance,
            runner_up_family=second_family,
            runner_up_subtype=self.subtypes_[second_family],
            runner_up_distance=second_distance,
            family_scores=family_scores,
        )


def default_clean_cases(num_samples=256):
    return [
        ("CW", "CW", cw(num_samples)),
        ("LFM", "LFM", lfm(num_samples, 0.30)),
        ("BIPHASE", "BARKER13", barker13(num_samples)),
        ("BIPHASE", "GENERIC_BIPHASE", biphase(num_samples)),
        ("POLYPHASE", "FRANK16_LIKE", polyphase(num_samples)),
    ]
