import copy
import math
import statistics


class ProbabilisticMultipleHypothesisAssociator:
    """Beam-pruned probabilistic multiple-hypothesis PDW associator.

    Each PDW branches into:
      * association with each existing emitter track,
      * birth of a new emitter track, or
      * clutter / false-alarm.

    Existing-track likelihoods combine Gaussian feature likelihoods, a
    categorical modulation likelihood, and a timing likelihood that explicitly
    allows missed pulses through integer PRI multiples. Global hypotheses
    accumulate log posterior weight and are beam-pruned for tractability.

    This is still a compact experimental implementation: beam pruning replaces
    exhaustive hypothesis-tree retention and the likelihood parameters are
    engineering models that will later need calibration from measured data.
    """

    def __init__(
        self,
        frequency_sigma_hz,
        pulse_width_sigma_s,
        amplitude_sigma_db,
        timing_sigma_s,
        beam_width=20,
        max_emitters=4,
        birth_probability=0.03,
        clutter_probability=0.005,
        modulation_match_probability=0.98,
        missed_pulse_probability=0.08,
        max_pri_multiple=4,
    ):
        self.frequency_sigma_hz = frequency_sigma_hz
        self.pulse_width_sigma_s = pulse_width_sigma_s
        self.amplitude_sigma_db = amplitude_sigma_db
        self.timing_sigma_s = timing_sigma_s
        self.beam_width = beam_width
        self.max_emitters = max_emitters
        self.birth_probability = birth_probability
        self.clutter_probability = clutter_probability
        self.modulation_match_probability = modulation_match_probability
        self.missed_pulse_probability = missed_pulse_probability
        self.max_pri_multiple = max_pri_multiple

    @staticmethod
    def _safe_log(value):
        return math.log(max(value, 1e-300))

    @staticmethod
    def _gaussian_log_kernel(residual, sigma):
        """Dimensionless Gaussian log likelihood ratio about zero residual."""
        if sigma <= 0.0:
            raise ValueError("Likelihood sigma must be positive")
        z = residual / sigma
        return -0.5 * z * z

    @staticmethod
    def _logsumexp(values):
        if not values:
            return float("-inf")
        maximum = max(values)
        if not math.isfinite(maximum):
            return maximum
        return maximum + math.log(sum(math.exp(v - maximum) for v in values))

    @staticmethod
    def _new_candidate(candidate_id, pdw):
        return {
            "candidate_id": candidate_id,
            "pdws": [pdw],
            "mean_frequency_hz": pdw.frequency_hz,
            "mean_pulse_width_s": pdw.pulse_width_s,
            "mean_amplitude_dbfs": pdw.amplitude_dbfs,
            "dominant_modulation": pdw.modulation_type,
            "estimated_pri_s": None,
        }

    @staticmethod
    def _update_candidate(candidate):
        pdws = candidate["pdws"]
        candidate["mean_frequency_hz"] = sum(p.frequency_hz for p in pdws) / len(pdws)
        candidate["mean_pulse_width_s"] = sum(p.pulse_width_s for p in pdws) / len(pdws)
        candidate["mean_amplitude_dbfs"] = sum(p.amplitude_dbfs for p in pdws) / len(pdws)

        modulation_counts = {}
        for pdw in pdws:
            modulation_counts[pdw.modulation_type] = modulation_counts.get(pdw.modulation_type, 0) + 1
        candidate["dominant_modulation"] = max(modulation_counts, key=modulation_counts.get)

        if len(pdws) >= 2:
            intervals = [
                pdws[index].toa_s - pdws[index - 1].toa_s
                for index in range(1, len(pdws))
            ]
            candidate["estimated_pri_s"] = statistics.median(intervals)
        else:
            candidate["estimated_pri_s"] = None

    def _modulation_log_likelihood(self, pdw, candidate):
        if pdw.modulation_type == "UNKNOWN" or candidate["dominant_modulation"] == "UNKNOWN":
            return self._safe_log(0.5)

        if pdw.modulation_type == candidate["dominant_modulation"]:
            return self._safe_log(self.modulation_match_probability)

        return self._safe_log(1.0 - self.modulation_match_probability)

    def _timing_log_likelihood(self, pdw, candidate):
        pri_s = candidate["estimated_pri_s"]
        if pri_s is None:
            return 0.0

        dt_s = pdw.toa_s - candidate["pdws"][-1].toa_s
        if dt_s <= 0.0:
            return self._safe_log(1e-12)

        terms = []
        p_miss = min(max(self.missed_pulse_probability, 1e-6), 0.95)

        for multiple in range(1, self.max_pri_multiple + 1):
            # m=1 means no missed pulse; m=2 means one missed pulse, etc.
            missed = multiple - 1
            if multiple < self.max_pri_multiple:
                prior = (1.0 - p_miss) * (p_miss ** missed)
            else:
                # Fold the remaining geometric tail into the final allowed multiple.
                prior = p_miss ** missed

            residual_s = dt_s - multiple * pri_s
            terms.append(
                self._safe_log(prior)
                + self._gaussian_log_kernel(residual_s, self.timing_sigma_s)
            )

        return self._logsumexp(terms)

    def _track_log_likelihood(self, pdw, candidate):
        log_likelihood = 0.0
        log_likelihood += self._gaussian_log_kernel(
            pdw.frequency_hz - candidate["mean_frequency_hz"],
            self.frequency_sigma_hz,
        )
        log_likelihood += self._gaussian_log_kernel(
            pdw.pulse_width_s - candidate["mean_pulse_width_s"],
            self.pulse_width_sigma_s,
        )
        log_likelihood += self._gaussian_log_kernel(
            pdw.amplitude_dbfs - candidate["mean_amplitude_dbfs"],
            self.amplitude_sigma_db,
        )
        log_likelihood += self._modulation_log_likelihood(pdw, candidate)
        log_likelihood += self._timing_log_likelihood(pdw, candidate)
        return log_likelihood

    @staticmethod
    def _signature(hypothesis):
        candidate_signature = tuple(
            tuple(p.pdw_id for p in candidate["pdws"])
            for candidate in hypothesis["candidates"]
        )
        clutter_signature = tuple(p.pdw_id for p in hypothesis["clutter_pdws"])
        return candidate_signature, clutter_signature

    @staticmethod
    def _candidate_ids(candidate):
        return {p.pdw_id for p in candidate["pdws"]}

    def _normalize_weights(self, hypotheses):
        if not hypotheses:
            return

        log_weights = [h["log_weight"] for h in hypotheses]
        log_norm = self._logsumexp(log_weights)
        for hypothesis in hypotheses:
            hypothesis["probability"] = math.exp(hypothesis["log_weight"] - log_norm)
            # Keep the existing display field. Lower score is better.
            hypothesis["score"] = -hypothesis["log_weight"]

    def associate(self, pdws):
        hypotheses = [
            {
                "log_weight": 0.0,
                "score": 0.0,
                "candidates": [],
                "clutter_pdws": [],
                "history": [],
            }
        ]

        p_birth = min(max(self.birth_probability, 1e-9), 0.95)
        p_clutter = min(max(self.clutter_probability, 1e-9), 0.95)
        p_existing = max(1.0 - p_birth - p_clutter, 1e-9)

        for pdw in pdws:
            branches = []

            for hypothesis in hypotheses:
                num_candidates = len(hypothesis["candidates"])

                if num_candidates > 0:
                    log_assignment_prior = self._safe_log(p_existing / num_candidates)
                    for candidate_index, candidate in enumerate(hypothesis["candidates"]):
                        branch = copy.deepcopy(hypothesis)
                        branch_candidate = branch["candidates"][candidate_index]
                        log_increment = log_assignment_prior + self._track_log_likelihood(
                            pdw,
                            candidate,
                        )
                        branch_candidate["pdws"].append(pdw)
                        self._update_candidate(branch_candidate)
                        branch["log_weight"] += log_increment
                        branch["history"].append(
                            (pdw.pdw_id, f"C{branch_candidate['candidate_id']}", log_increment)
                        )
                        branches.append(branch)

                if num_candidates < self.max_emitters:
                    branch = copy.deepcopy(hypothesis)
                    candidate_id = len(branch["candidates"]) + 1
                    branch["candidates"].append(self._new_candidate(candidate_id, pdw))
                    log_increment = self._safe_log(p_birth)
                    branch["log_weight"] += log_increment
                    branch["history"].append(
                        (pdw.pdw_id, f"BIRTH C{candidate_id}", log_increment)
                    )
                    branches.append(branch)

                branch = copy.deepcopy(hypothesis)
                log_increment = self._safe_log(p_clutter)
                branch["clutter_pdws"].append(pdw)
                branch["log_weight"] += log_increment
                branch["history"].append((pdw.pdw_id, "CLUTTER", log_increment))
                branches.append(branch)

            unique = {}
            for branch in branches:
                signature = self._signature(branch)
                if signature not in unique or branch["log_weight"] > unique[signature]["log_weight"]:
                    unique[signature] = branch

            hypotheses = sorted(
                unique.values(),
                key=lambda item: item["log_weight"],
                reverse=True,
            )[: self.beam_width]

        self._normalize_weights(hypotheses)
        return hypotheses

    def association_marginals(self, hypotheses):
        """Coarse per-PDW emitter-family posterior mass across retained hypotheses."""
        if not hypotheses:
            return {}

        reference_candidates = hypotheses[0]["candidates"]
        reference_sets = {
            candidate["candidate_id"]: self._candidate_ids(candidate)
            for candidate in reference_candidates
        }

        pdw_ids = sorted(
            {
                pdw.pdw_id
                for hypothesis in hypotheses
                for candidate in hypothesis["candidates"]
                for pdw in candidate["pdws"]
            }
            | {
                pdw.pdw_id
                for hypothesis in hypotheses
                for pdw in hypothesis.get("clutter_pdws", [])
            }
        )

        marginals = {
            pdw_id: {
                **{candidate_id: 0.0 for candidate_id in reference_sets},
                "OTHER": 0.0,
            }
            for pdw_id in pdw_ids
        }

        for hypothesis in hypotheses:
            weight = hypothesis.get("probability", 0.0)
            for candidate in hypothesis["candidates"]:
                candidate_ids = self._candidate_ids(candidate)
                best_reference = None
                best_overlap = 0

                for reference_id, reference_ids in reference_sets.items():
                    overlap = len(candidate_ids & reference_ids)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_reference = reference_id

                label = best_reference if best_reference is not None else "OTHER"
                for pdw_id in candidate_ids:
                    marginals[pdw_id][label] += weight

            for pdw in hypothesis.get("clutter_pdws", []):
                marginals[pdw.pdw_id]["OTHER"] += weight

        for distribution in marginals.values():
            total = sum(distribution.values())
            if total > 0.0:
                for label in distribution:
                    distribution[label] /= total

        return marginals

    def reference_track_membership(self, hypotheses):
        """Posterior co-association with each best-hypothesis reference track."""
        if not hypotheses:
            return {}

        reference_candidates = hypotheses[0]["candidates"]
        reference = {
            candidate["candidate_id"]: {
                "anchor_pdw_id": candidate["pdws"][0].pdw_id,
                "reference_pdw_ids": self._candidate_ids(candidate),
            }
            for candidate in reference_candidates
        }

        membership = {
            pdw_id: {candidate_id: 0.0 for candidate_id in reference}
            for candidate in reference_candidates
            for pdw_id in self._candidate_ids(candidate)
        }

        for hypothesis in hypotheses:
            weight = hypothesis.get("probability", 0.0)
            hypothesis_sets = [self._candidate_ids(c) for c in hypothesis["candidates"]]

            for candidate_id, info in reference.items():
                anchor = info["anchor_pdw_id"]
                anchor_set = next((ids for ids in hypothesis_sets if anchor in ids), set())

                for pdw_id in info["reference_pdw_ids"]:
                    if pdw_id in anchor_set:
                        membership[pdw_id][candidate_id] += weight

        return membership
