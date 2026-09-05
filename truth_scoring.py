from pulse_timing import generate_pulse_toas


class SimulationTruthScorer:
    """Score ESM output against simulator truth without feeding truth to the ESM.

    A scenario may contain several waveform/mode segments from one physical
    emitter. Truth scoring therefore distinguishes physical emitter identity
    from the number of signal segments used to generate the RF.
    """

    def __init__(self, sample_rate_hz, toa_tolerance_samples=2):
        self.sample_rate_hz = sample_rate_hz
        self.toa_tolerance_s = toa_tolerance_samples / sample_rate_hz

    @staticmethod
    def _physical_emitters(scenario):
        physical_ids = []
        for segment in scenario.emitters:
            physical_id = segment.get("physical_emitter_id", segment.get("name", "UNKNOWN"))
            if physical_id not in physical_ids:
                physical_ids.append(physical_id)
        return physical_ids

    @staticmethod
    def _true_modes(scenario):
        modes = []
        for segment in scenario.emitters:
            mode = segment.get("mode", "UNSPECIFIED")
            if mode != "UNSPECIFIED" and mode not in modes:
                modes.append(mode)
        return modes

    def _truth_pulses(self, scenario):
        physical_ids = self._physical_emitters(scenario)
        physical_index = {physical_id: index + 1 for index, physical_id in enumerate(physical_ids)}

        truth = []
        for segment_index, emitter in enumerate(scenario.emitters, start=1):
            emitter_name = emitter.get("name", f"Segment {segment_index}")
            physical_id = emitter.get("physical_emitter_id", emitter_name)
            mode = emitter.get("mode", "UNSPECIFIED")
            toas = generate_pulse_toas(
                start_delay_s=emitter["start_delay_s"],
                pri_s=emitter["pri_s"],
                num_pulses=emitter["num_pulses"],
                pri_jitter_s=emitter.get("pri_jitter_s", 0.0),
                random_seed=emitter.get("random_seed", 0),
            )
            for pulse_index, toa_s in enumerate(toas, start=1):
                truth.append(
                    {
                        "physical_emitter_index": physical_index[physical_id],
                        "physical_emitter_id": physical_id,
                        "segment_index": segment_index,
                        "segment_name": emitter_name,
                        "mode": mode,
                        "pulse_index": pulse_index,
                        "toa_s": toa_s,
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
                if error <= self.toa_tolerance_s and (best_error is None or error < best_error):
                    best_index = truth_index
                    best_error = error

            if best_index is not None:
                truth = truth_pulses[best_index]
                pdw_truth[pdw.pdw_id] = truth["physical_emitter_index"]
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

    @staticmethod
    def _best_label_mapping(pdw_truth, candidate_membership, candidate_ids):
        """Map each inferred track to the physical emitter owning most of its PDWs.

        Mapping is intentionally many-to-one: two inferred tracks can both map
        to one real radar. This allows truth scoring to distinguish correct
        physical-origin association from erroneous track fragmentation during a
        radar mode change.
        """
        mapping = {}

        for candidate_id in candidate_ids:
            counts = {}
            for pdw_id, assigned_candidate in candidate_membership.items():
                if assigned_candidate != candidate_id:
                    continue
                truth_id = pdw_truth.get(pdw_id)
                if truth_id is not None:
                    counts[truth_id] = counts.get(truth_id, 0) + 1

            if counts:
                mapping[candidate_id] = max(counts, key=counts.get)

        correct = sum(
            1
            for pdw_id, truth_id in pdw_truth.items()
            if truth_id is not None
            and pdw_id in candidate_membership
            and mapping.get(candidate_membership[pdw_id]) == truth_id
        )
        return mapping, correct

    def score(self, scenario, pdws, hypotheses):
        physical_emitters = self._physical_emitters(scenario)
        true_modes = self._true_modes(scenario)
        truth_pulses = self._truth_pulses(scenario)
        pdw_truth, matched, missed = self._match_detections_to_truth(pdws, truth_pulses)

        false = len(pdws) - matched
        true_count = len(truth_pulses)
        true_emitter_count = len(physical_emitters)
        recall = matched / true_count if true_count else 1.0
        precision = matched / len(pdws) if pdws else (1.0 if true_count == 0 else 0.0)

        if not hypotheses:
            return self._result(
                true_emitter_count, 0, true_count, len(pdws), matched, missed,
                false, recall, precision, 0, matched, {}, False,
                physical_emitters, true_modes, len(scenario.emitters),
            )

        best = hypotheses[0]
        membership = self._candidate_membership(best)
        candidate_ids = [c["candidate_id"] for c in best["candidates"]]
        mapping, correct = self._best_label_mapping(pdw_truth, membership, candidate_ids)
        estimated = len(best["candidates"])

        exact = (
            estimated == true_emitter_count
            and correct == matched
            and false == 0
            and missed == 0
        )

        return self._result(
            true_emitter_count, estimated, true_count, len(pdws), matched, missed,
            false, recall, precision, correct, matched, mapping, exact,
            physical_emitters, true_modes, len(scenario.emitters),
        )

    @staticmethod
    def _result(
        true_emitters,
        estimated,
        true_pulses,
        detected,
        matched,
        missed,
        false,
        recall,
        precision,
        correct,
        total,
        mapping,
        exact,
        physical_emitters,
        true_modes,
        signal_segments,
    ):
        return {
            "true_emitters": true_emitters,
            "estimated_emitters": estimated,
            "signal_segments": signal_segments,
            "physical_emitter_names": physical_emitters,
            "true_modes": true_modes,
            "true_pulses": true_pulses,
            "detected_pulses": detected,
            "matched_detections": matched,
            "missed_pulses": missed,
            "false_detections": false,
            "detection_recall": recall,
            "detection_precision": precision,
            "association_correct": correct,
            "association_total": total,
            "association_accuracy": correct / total if total else 1.0,
            "emitter_count_match": estimated == true_emitters,
            "track_fragmentation": max(0, estimated - true_emitters),
            "best_hypothesis_match": exact,
            "candidate_truth_mapping": mapping,
        }


def print_truth_score(score):
    print("SIMULATION TRUTH / SCORING")
    print("==========================")
    print("Truth is used only here for evaluation; it is not available to the ESM algorithms.")
    print()
    print(f"True physical emitters : {score['true_emitters']}")
    print(f"Signal/mode segments   : {score['signal_segments']}")
    if score["true_modes"]:
        print(f"True modes             : {', '.join(score['true_modes'])}")
    print(f"Estimated emitters     : {score['estimated_emitters']}")
    print(f"Emitter count          : {'MATCH' if score['emitter_count_match'] else 'MISMATCH'}")
    print(f"Track fragmentation    : {score['track_fragmentation']}")
    print()
    print(f"True pulses            : {score['true_pulses']}")
    print(f"Detected pulses        : {score['detected_pulses']}")
    print(f"Missed pulses          : {score['missed_pulses']}")
    print(f"False detections       : {score['false_detections']}")
    print(f"Detection recall       : {100.0 * score['detection_recall']:.1f}%")
    print(f"Detection precision    : {100.0 * score['detection_precision']:.1f}%")
    print()
    print(
        f"Physical-origin assoc. : {score['association_correct']} / "
        f"{score['association_total']} correct"
    )
    print(f"Association accuracy   : {100.0 * score['association_accuracy']:.1f}%")
    print(
        f"Best hypothesis        : "
        f"{'MATCH' if score['best_hypothesis_match'] else 'NOT EXACT MATCH'}"
    )

    if score["candidate_truth_mapping"]:
        mapping_text = ", ".join(
            f"C{candidate}->Truth E{truth}"
            for candidate, truth in sorted(score["candidate_truth_mapping"].items())
        )
        print(f"Best label mapping     : {mapping_text}")
    print()
