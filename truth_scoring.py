from itertools import permutations

from pulse_timing import generate_pulse_toas


class SimulationTruthScorer:
    """Score ESM output against simulator truth without feeding truth to the ESM."""

    def __init__(self, sample_rate_hz, toa_tolerance_samples=2):
        self.sample_rate_hz = sample_rate_hz
        self.toa_tolerance_s = toa_tolerance_samples / sample_rate_hz

    def _truth_pulses(self, scenario):
        truth = []
        for emitter_index, emitter in enumerate(scenario.emitters, start=1):
            emitter_name = emitter.get("name", f"Emitter {emitter_index}")
            toas = generate_pulse_toas(
                start_delay_s=emitter["start_delay_s"],
                pri_s=emitter["pri_s"],
                num_pulses=emitter["num_pulses"],
                pri_jitter_s=emitter.get("pri_jitter_s", 0.0),
                random_seed=emitter.get("random_seed", 0),
            )
            for pulse_index, toa_s in enumerate(toas, start=1):
                truth.append({
                    "emitter_index": emitter_index,
                    "emitter_name": emitter_name,
                    "pulse_index": pulse_index,
                    "toa_s": toa_s,
                })
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
                if error <= self.toa_tolerance_s and (best_error is None or error < best_error):
                    best_index = truth_index
                    best_error = error
            if best_index is not None:
                pdw_truth[pdw.pdw_id] = truth_pulses[best_index]["emitter_index"]
                unmatched_truth.remove(best_index)
                matched += 1
            else:
                pdw_truth[pdw.pdw_id] = None
        return pdw_truth, matched, len(unmatched_truth)

    @staticmethod
    def _candidate_membership(best_hypothesis):
        membership = {}
        for candidate in best_hypothesis["candidates"]:
            for pdw in candidate["pdws"]:
                membership[pdw.pdw_id] = candidate["candidate_id"]
        return membership

    def _best_label_mapping(self, pdw_truth, candidate_membership, num_truth_emitters, candidate_ids):
        if not candidate_ids or num_truth_emitters == 0:
            return {}, 0
        truth_ids = list(range(1, num_truth_emitters + 1))
        best_mapping, best_correct = {}, -1
        if len(candidate_ids) <= len(truth_ids):
            mappings = (dict(zip(candidate_ids, p)) for p in permutations(truth_ids, len(candidate_ids)))
        else:
            mappings = (
                dict(zip(selected, truth_perm))
                for selected in permutations(candidate_ids, len(truth_ids))
                for truth_perm in permutations(truth_ids)
            )
        for mapping in mappings:
            correct = sum(
                1 for pdw_id, truth_id in pdw_truth.items()
                if truth_id is not None
                and pdw_id in candidate_membership
                and mapping.get(candidate_membership[pdw_id]) == truth_id
            )
            if correct > best_correct:
                best_mapping, best_correct = mapping, correct
        return best_mapping, max(best_correct, 0)

    def score(self, scenario, pdws, hypotheses):
        truth_pulses = self._truth_pulses(scenario)
        pdw_truth, matched, missed = self._match_detections_to_truth(pdws, truth_pulses)
        false = len(pdws) - matched
        true_count = len(truth_pulses)
        recall = matched / true_count if true_count else 1.0
        precision = matched / len(pdws) if pdws else (1.0 if true_count == 0 else 0.0)

        if not hypotheses:
            return self._result(len(scenario.emitters), 0, true_count, len(pdws), matched, missed, false, recall, precision, 0, matched, {}, False)

        best = hypotheses[0]
        membership = self._candidate_membership(best)
        candidate_ids = [c["candidate_id"] for c in best["candidates"]]
        mapping, correct = self._best_label_mapping(pdw_truth, membership, len(scenario.emitters), candidate_ids)
        estimated = len(best["candidates"])
        exact = estimated == len(scenario.emitters) and correct == matched and false == 0 and missed == 0
        return self._result(len(scenario.emitters), estimated, true_count, len(pdws), matched, missed, false, recall, precision, correct, matched, mapping, exact)

    @staticmethod
    def _result(true_emitters, estimated, true_pulses, detected, matched, missed, false, recall, precision, correct, total, mapping, exact):
        return {
            "true_emitters": true_emitters, "estimated_emitters": estimated,
            "true_pulses": true_pulses, "detected_pulses": detected,
            "matched_detections": matched, "missed_pulses": missed,
            "false_detections": false, "detection_recall": recall,
            "detection_precision": precision, "association_correct": correct,
            "association_total": total,
            "association_accuracy": correct / total if total else 1.0,
            "emitter_count_match": estimated == true_emitters,
            "best_hypothesis_match": exact, "candidate_truth_mapping": mapping,
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
    print(f"PDW association      : {score['association_correct']} / {score['association_total']} correct")
    print(f"Association accuracy : {100.0 * score['association_accuracy']:.1f}%")
    print(f"Best hypothesis      : {'MATCH' if score['best_hypothesis_match'] else 'NOT EXACT MATCH'}")
    if score["candidate_truth_mapping"]:
        mapping_text = ", ".join(
            f"C{candidate}->Truth E{truth}"
            for candidate, truth in sorted(score["candidate_truth_mapping"].items())
        )
        print(f"Best label mapping   : {mapping_text}")
    print()
