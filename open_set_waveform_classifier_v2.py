from dataclasses import dataclass

import numpy as np

from hierarchical_waveform_classifier import HierarchicalWaveformClassifier


@dataclass(frozen=True)
class OpenSetWaveformClassificationV2:
    family: str
    candidate_family: str
    accepted: bool
    rejection_reason: str
    confidence: float
    occupied_bandwidth_hz: float
    family_scores: dict
    physics_scores: dict
    coded_reconstruction_ratio: float
    coded_nearest_ratio: float
    smooth_residual_ratio: float


class OpenSetWaveformClassifierV2:
    """Open-set waveform classifier using model adequacy, not nearest class alone.

    CW/LFM candidates must satisfy a smooth-model residual envelope learned from
    training examples. Coded candidates must satisfy both PCA reconstruction-error
    and nearest-training-neighbour envelopes in normalized spectral-feature space.
    """

    def __init__(self, sample_rate_hz=40_000_000.0):
        self.base = HierarchicalWaveformClassifier(sample_rate_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self._smooth_limits = {}
        self._coded_recon_limits = {}
        self._coded_nn_limits = {}
        self._coded_training_z = {}

    @staticmethod
    def _quantile_limit(values, q=0.98, margin=1.20):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.inf
        return margin * float(np.quantile(values, q))

    def _smooth_model_metrics(self, samples):
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 16:
            return {"CW": np.inf, "LFM": np.inf}

        edge = max(2, int(round(0.04 * len(x))))
        if len(x) - 2 * edge >= 12:
            x = x[edge:-edge]

        q = np.angle(x[1:] * np.conj(x[:-1]))
        n = np.arange(len(q), dtype=float)

        # Robustly ignore sparse phase discontinuities when judging smooth models.
        med = float(np.median(q))
        r0 = np.angle(np.exp(1j * (q - med)))
        abs_r0 = np.abs(r0)
        central = r0[abs_r0 <= np.quantile(abs_r0, 0.70)]
        mad = float(np.median(np.abs(central - np.median(central)))) if len(central) else 0.0
        noise = max(1.4826 * mad, 1e-4)
        jump_mask = abs_r0 > max(5.0 * noise, 0.35)
        keep = ~jump_mask
        if np.count_nonzero(keep) < 8:
            keep = np.ones_like(jump_mask, dtype=bool)

        qs = q[keep]
        ns = n[keep]

        cw_fit = float(np.mean(qs))
        cw_resid = np.angle(np.exp(1j * (qs - cw_fit)))
        cw_rms = float(np.sqrt(np.mean(cw_resid ** 2)))

        a = np.column_stack((np.ones_like(ns), ns))
        coeff, _, _, _ = np.linalg.lstsq(a, qs, rcond=None)
        lfm_resid = np.angle(np.exp(1j * (qs - a @ coeff)))
        lfm_rms = float(np.sqrt(np.mean(lfm_resid ** 2)))

        # Normalize by the pulse's own robust phase-noise scale. Unknown smooth
        # structures such as NLFM/stepped frequency should leave excess residual.
        return {
            "CW": cw_rms / max(noise, 1e-4),
            "LFM": lfm_rms / max(noise, 1e-4),
        }

    def _coded_feature_metrics(self, samples, label=None):
        feature, _ = self.base.coded._spectral_feature(samples)
        if feature is None:
            return np.inf, np.inf

        centered = feature - self.base.coded.mean_
        projected = centered @ self.base.coded.components_.T
        reconstructed = projected @ self.base.coded.components_
        recon_error = float(np.linalg.norm(centered - reconstructed))

        z = self.base.coded._project_matrix(feature[None, :])[0]
        if label in self._coded_training_z and len(self._coded_training_z[label]):
            distances = np.linalg.norm(self._coded_training_z[label] - z, axis=1)
            nearest = float(np.min(distances))
        else:
            nearest = np.inf
        return recon_error, nearest

    def fit(self, samples, labels):
        self.base.fit_coded(samples, labels)

        smooth_values = {"CW": [], "LFM": []}
        coded_recon = {"BIPHASE": [], "POLYPHASE": []}
        coded_z = {"BIPHASE": [], "POLYPHASE": []}

        for waveform, label in zip(samples, labels):
            if label in smooth_values:
                metrics = self._smooth_model_metrics(waveform)
                smooth_values[label].append(metrics[label])

            if label in coded_recon:
                feature, _ = self.base.coded._spectral_feature(waveform)
                if feature is None:
                    continue
                centered = feature - self.base.coded.mean_
                projected = centered @ self.base.coded.components_.T
                reconstructed = projected @ self.base.coded.components_
                coded_recon[label].append(float(np.linalg.norm(centered - reconstructed)))
                z = self.base.coded._project_matrix(feature[None, :])[0]
                coded_z[label].append(z)

        self._smooth_limits = {
            label: self._quantile_limit(values, q=0.98, margin=1.25)
            for label, values in smooth_values.items()
        }
        self._coded_recon_limits = {
            label: self._quantile_limit(values, q=0.98, margin=1.25)
            for label, values in coded_recon.items()
        }
        self._coded_training_z = {
            label: np.vstack(values) if values else np.empty((0, self.base.coded.n_components))
            for label, values in coded_z.items()
        }

        # Leave-one-neighbour distances provide an empirical manifold envelope.
        for label, z in self._coded_training_z.items():
            nn = []
            for i in range(len(z)):
                if len(z) < 2:
                    continue
                distances = np.linalg.norm(z - z[i], axis=1)
                distances[i] = np.inf
                nn.append(float(np.min(distances)))
            self._coded_nn_limits[label] = self._quantile_limit(nn, q=0.98, margin=1.35)
        return self

    def classify(self, samples):
        result = self.base.classify(samples)
        candidate = result.family
        accepted = True
        reason = ""
        recon_ratio = 0.0
        nn_ratio = 0.0
        smooth_ratio = 0.0

        if candidate in ("CW", "LFM"):
            metric = self._smooth_model_metrics(samples)[candidate]
            limit = max(self._smooth_limits.get(candidate, np.inf), 1e-9)
            smooth_ratio = metric / limit
            if smooth_ratio > 1.0:
                accepted = False
                reason = "smooth model residual too large"

        elif candidate in ("BIPHASE", "POLYPHASE"):
            recon, nearest = self._coded_feature_metrics(samples, candidate)
            recon_limit = max(self._coded_recon_limits.get(candidate, np.inf), 1e-9)
            nn_limit = max(self._coded_nn_limits.get(candidate, np.inf), 1e-9)
            recon_ratio = recon / recon_limit
            nn_ratio = nearest / nn_limit
            if recon_ratio > 1.0:
                accepted = False
                reason = "coded PCA reconstruction error too large"
            elif nn_ratio > 1.0:
                accepted = False
                reason = "outside coded training manifold"
        else:
            accepted = False
            reason = "no supported family"

        return OpenSetWaveformClassificationV2(
            family=candidate if accepted else "UNKNOWN",
            candidate_family=candidate,
            accepted=accepted,
            rejection_reason=reason,
            confidence=result.confidence,
            occupied_bandwidth_hz=result.occupied_bandwidth_hz,
            family_scores=result.family_scores,
            physics_scores=result.physics_scores,
            coded_reconstruction_ratio=recon_ratio,
            coded_nearest_ratio=nn_ratio,
            smooth_residual_ratio=smooth_ratio,
        )
