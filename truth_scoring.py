from itertools import permutations


class SimulationTruthScorer:
    """Score ESM output against simulator truth without feeding truth to the ESM.

    Truth is reconstructed only after detection and association from the known
    scenario definition. The ESM processing chain never receives emitter names
    or true pulse associations.
    """

    def __init__(self, sample_rate_hz, toa_tolerance_samples=2):
        self.sample_rate_hz = sample_rate_hz
        self.toa_tolerance_s = toa_tolerance_samples / sample_rate_hz

    def _truth_pulses(self, scenario):
        truth = []
        for emitter_index, emitter in enumerate(scenario.emitters, start=1):
            emitter_name = emitter.get("name", f"Emitter {emitter_index}")
            for pulse_index in range(emitter["num_pulses"]):
                truth.append(
                    {
                        "emitter_index": emitter_index,
                        "emitter_name": emitter_name,
                        "pulse_index": pulse_index + 1,
                        "toa_s": emitter["start_delay_s"] + pulse_index * emitter["pri_s"],
                    }
                )
        return sorted(truth, key=lambda item: item["toa_s"])

    def _match_detections_to_truth(self, pdws, truth_pulses):
        unmatched_truth = set(range(len(truth_pulses)))
        pdw_truth = {}
        matched = 0

        for pdw in pdws:
            best_index = None
            best_error = None

            for truth_index in unmatched_truth:
                error = abs(pdw.toa_s - truth_pulses[truth_index]["toa_s"])
                if error <= self.toa_tolerance_s:
                    if best_error is None or error < best_error:
                        best_index = truth_index
                        best_error = error

            if best_index is not None:
                truth = truth_pulses[best_index]
                pdw_truth[pdw.pdw_id] = truth["emitter_index"]
                unmatched_truth.remove(best_index)
                matched += 1
            else:
                pdw_truth[pdw.pdw_id] = None

        return pdw_truth, matched, len(unmatched_truth)

    @staticmethod
    def _candidate_membership(best_hypothesis):
        membership = {}
        for candidate in best_hypothesis["candidates"]:
            candidate_id = candidate["candidate_id"]
            for pdw in candidate["pdws"]:
                membership[pdw.pdw_id] = candidate_id
        return membership

    def _best_label_mapping(self, pdw_truth, candidate_membership, num_truth_emitters, candidate_ids):
        """Find label-invariant candidate-to-truth mapping maximizing correct PDWs."""
        if not candidate_ids or num_truth_emitters == 0:
            return {}, 0

        truth_ids = list(range(1, num_truth_emitters + 1))
        best_mapping = {}
        best_correct = -1

        # Candidate count is intentionally small in this experimental MHT.
        # Map as many candidates as possible one-to-one to real emitters.
        if len(candidate_ids) <= len(truth_ids):
            for truth_perm in permutations(truth_ids, len(candidate_ids)):
                mapping = dict(zip(candidate_ids, truth_perm))
                correct = sum(
                    1
                    for pdw_id, truth_id in pdw_truth.items()
                    if truth_id is not None
                    and pdw_id in candidate_membership
                    and mapping.get(candidate_membership[pdw_id]) == truth_id
                )
                if correct > best_correct:
                    best_correct = correct
                    best_mapping = mapping
        else:
            # More inferred candidates than true emitters: choose which candidates
            # receive distinct truth labels; the remainder are treated as splits.
            for selected_candidates in permutations(candidate_ids, len(truth_ids)):
                for truth_perm in permutations(truth_ids):
                    mapping = dict(zip(selected_candidates, truth_perm))
                    correct = sum(
                        1
                        for pdw_id, truth_id in pdw_truth.items()
                        if truth_id is not None
                        and pdw_id in candidate_membership
                        and mapping.get(candidate_membership[pdw_id]) == truth_id
                    )
                    if correct > best_correct:
                        best_correct = correct
                        best_mapping = mapping

        return best_mapping, max(best_correct, 0)

    def score(self, scenario, pdws, hypotheses):
        truth_pulses = self._truth_pulses(scenario)
        pdw_truth, matched_detections, missed_truth = self._match_detections_to_truth(
            pdws,
            truth_pulses,
        )

        false_detections = len(pdws) - matched_detections
        true_pulse_count = len(truth_pulses)
        detection_recall = matched_detections / true_pulse_count if true_pulse_count else 1.0
        detection_precision = matched_detections / len(pdws) if pdws else (1.0 if true_pulse_count == 0 else 0.0)

        if not hypotheses:
            return {
                "true_emitters": len(scenario.emitters),
                "estimated_emitters": 0,
                "true_pulses": true_pulse_count,
                "detected_pulses": len(pdws),
                "matched_detections": matched_detections,
                "missed_pulses": missed_truth,
                "false_detections": false_detections,
                "detection_recall": detection_recall,
                "detection_precision": detection_precision,
                "association_correct": 0,
                "association_total": matched_detections,
                "association_accuracy": 0.0 if matched_detections else 1.0,
                "emitter_count_match": False,
                "best_hypothesis_match": False,
                "candidate_truth_mapping": {},
            }

        best_hypothesis = hypotheses[0]
        candidate_membership = self._candidate_membership(best_hypothesis)
        candidate_ids = [c["candidate_id"] for c in best_hypothesis["candidates"]]

        mapping, association_correct = self._best_label_mapping(
            pdw_truth,
            candidate_membership,
            len(scenario.emitters),
            candidate_ids,
        )

        association_total = matched_detections
        association_accuracy = (
            association_correct / association_total if association_total else 1.0
        )
        estimated_emitters = len(best_hypothesis["candidates"])
        emitter_count_match = estimated_emitters == len(scenario.emitters)
        best_hypothesis_match = (
            emitter_count_match
            and association_correct == association_total
            and false_detections == 0
            and missed_truth == 0
        )

        return {
            "true_emitters": len(scenario.emitters),
            "estimated_emitters": estimated_emitters,
            "true_pulses": true_pulse_count,
            "detected_pulses": len(pdws),
            "matched_detections": matched_detections,
            "missed_pulses": missed_truth,
            "false_detections": false_detections,
            "detection_recall": detection_recall,
            "detection_precision": detection_precision,
            "association_correct": association_correct,
            "association_total": association_total,
            "association_accuracy": association_accuracy,
            "emitter_count_match": emitter_count_match,
            "best_hypothesis_match": best_hypothesis_match,
            "candidate_truth_mapping": mapping,
        }


def print_truth_score(score):
    print("SIMULATION TRUTH / SCORING")
    print("==========================")
    print("Truth is used only here for evaluation; it is not available to the ESM algorithms.")
    print()
    print(f"True emitters        : {score['true_emitters']}")
    print(f"Estimated emitters   : {score['estimated_emitters']}")
    print(f"Emitter count        : {'MATCH' if score['emitter_count_match'] else 'MISMATCH'}")
    print()
    print(f"True pulses          : {score['true_pulses']}")
    print(f"Detected pulses      : {score['detected_pulses']}")
    print(f"Missed pulses        : {score['missed_pulses']}")
    print(f"False detections     : {score['false_detections']}")
    print(f"Detection recall     : {100.0 * score['detection_recall']:.1f}%")
    print(f"Detection precision  : {100.0 * score['detection_precision']:.1f}%")
    print()
    print(
        f"PDW association      : {score['association_correct']} / "
        f"{score['association_total']} correct"
    )
    print(f"Association accuracy : {100.0 * score['association_accuracy']:.1f}%")
    print(
        f"Best hypothesis      : "
        f"{'MATCH' if score['best_hypothesis_match'] else 'NOT EXACT MATCH'}"
    )

    if score["candidate_truth_mapping"]:
        mapping_text = ", ".join(
            f"C{candidate}->Truth E{truth}"
            for candidate, truth in sorted(score["candidate_truth_mapping"].items())
        )
        print(f"Best label mapping   : {mapping_text}")
    print()
