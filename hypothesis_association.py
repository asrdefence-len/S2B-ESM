import copy
import math
import statistics


class MultipleHypothesisAssociator:
    """Small beam-search multiple-hypothesis PDW associator.

    Each arriving PDW can be assigned to any existing candidate emitter or can
    start a new candidate. The resulting global hypotheses are scored using
    measured PDW evidence only and the best N hypotheses are retained.

    This is intentionally experimental rather than a full operational MHT.
    Its purpose is to preserve early ambiguity so later evidence can repair
    initially plausible but incorrect associations.
    """

    def __init__(
        self,
        frequency_scale_hz,
        pulse_width_scale_s,
        amplitude_scale_db,
        timing_scale_s,
        beam_width=20,
        max_emitters=4,
        new_emitter_penalty=1.8,
        modulation_mismatch_penalty=2.0,
        max_pri_multiple=3,
    ):
        self.frequency_scale_hz = frequency_scale_hz
        self.pulse_width_scale_s = pulse_width_scale_s
        self.amplitude_scale_db = amplitude_scale_db
        self.timing_scale_s = timing_scale_s
        self.beam_width = beam_width
        self.max_emitters = max_emitters
        self.new_emitter_penalty = new_emitter_penalty
        self.modulation_mismatch_penalty = modulation_mismatch_penalty
        self.max_pri_multiple = max_pri_multiple

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
        for p in pdws:
            modulation_counts[p.modulation_type] = modulation_counts.get(p.modulation_type, 0) + 1
        candidate["dominant_modulation"] = max(modulation_counts, key=modulation_counts.get)

        if len(pdws) >= 2:
            intervals = [
                pdws[index].toa_s - pdws[index - 1].toa_s
                for index in range(1, len(pdws))
            ]
            candidate["estimated_pri_s"] = statistics.median(intervals)
        else:
            candidate["estimated_pri_s"] = None

    def _timing_cost(self, pdw, candidate):
        pri_s = candidate["estimated_pri_s"]
        if pri_s is None:
            return 0.0

        dt_s = pdw.toa_s - candidate["pdws"][-1].toa_s
        if dt_s <= 0.0:
            return 25.0

        timing_error_s = min(
            abs(dt_s - multiple * pri_s)
            for multiple in range(1, self.max_pri_multiple + 1)
        )
        return (timing_error_s / self.timing_scale_s) ** 2

    def _assignment_cost(self, pdw, candidate):
        frequency_cost = (
            (pdw.frequency_hz - candidate["mean_frequency_hz"])
            / self.frequency_scale_hz
        ) ** 2
        pulse_width_cost = (
            (pdw.pulse_width_s - candidate["mean_pulse_width_s"])
            / self.pulse_width_scale_s
        ) ** 2
        amplitude_cost = (
            (pdw.amplitude_dbfs - candidate["mean_amplitude_dbfs"])
            / self.amplitude_scale_db
        ) ** 2

        modulation_cost = 0.0
        if pdw.modulation_type != candidate["dominant_modulation"]:
            modulation_cost = self.modulation_mismatch_penalty

        timing_cost = self._timing_cost(pdw, candidate)

        return (
            frequency_cost
            + pulse_width_cost
            + amplitude_cost
            + modulation_cost
            + timing_cost
        )

    @staticmethod
    def _signature(hypothesis):
        """Canonical assignment signature used to remove duplicate branches."""
        return tuple(
            tuple(p.pdw_id for p in candidate["pdws"])
            for candidate in hypothesis["candidates"]
        )

    def associate(self, pdws):
        hypotheses = [
            {
                "score": 0.0,
                "candidates": [],
                "history": [],
            }
        ]

        for pdw in pdws:
            branches = []

            for hypothesis in hypotheses:
                # Branch 1..N: assign this PDW to each existing emitter candidate.
                for candidate_index, candidate in enumerate(hypothesis["candidates"]):
                    branch = copy.deepcopy(hypothesis)
                    cost = self._assignment_cost(pdw, candidate)
                    branch_candidate = branch["candidates"][candidate_index]
                    branch_candidate["pdws"].append(pdw)
                    self._update_candidate(branch_candidate)
                    branch["score"] += cost
                    branch["history"].append(
                        (pdw.pdw_id, branch_candidate["candidate_id"], cost)
                    )
                    branches.append(branch)

                # Additional branch: this PDW begins a previously unseen emitter.
                if len(hypothesis["candidates"]) < self.max_emitters:
                    branch = copy.deepcopy(hypothesis)
                    candidate_id = len(branch["candidates"]) + 1
                    branch["candidates"].append(
                        self._new_candidate(candidate_id, pdw)
                    )
                    branch["score"] += self.new_emitter_penalty
                    branch["history"].append(
                        (pdw.pdw_id, candidate_id, self.new_emitter_penalty)
                    )
                    branches.append(branch)

            # Remove duplicate assignment histories, then retain only the best beam.
            unique = {}
            for branch in branches:
                signature = self._signature(branch)
                if signature not in unique or branch["score"] < unique[signature]["score"]:
                    unique[signature] = branch

            hypotheses = sorted(
                unique.values(),
                key=lambda item: item["score"],
            )[: self.beam_width]

        # Convert costs to relative probabilities for display. They are model
        # probabilities, not calibrated real-world probabilities.
        if hypotheses:
            best_score = hypotheses[0]["score"]
            weights = [
                math.exp(-0.5 * min(h["score"] - best_score, 100.0))
                for h in hypotheses
            ]
            total_weight = sum(weights)
            for hypothesis, weight in zip(hypotheses, weights):
                hypothesis["probability"] = weight / total_weight

        return hypotheses
