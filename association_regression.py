import argparse
import time

from config import *
from scenarios import get_scenario, list_scenarios
from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator
from gated_fast_probabilistic_mht import GatedFastProbabilisticMultipleHypothesisAssociator
from truth_scoring import SimulationTruthScorer


def make_mht(cls):
    return cls(
        frequency_sigma_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
        pulse_width_sigma_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,
        amplitude_sigma_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
        timing_sigma_s=ASSOCIATION_TIMING_TOLERANCE_S,
        beam_width=MHT_BEAM_WIDTH,
        max_emitters=MHT_MAX_EMITTERS,
        birth_probability=PMHT_BIRTH_PROBABILITY,
        clutter_probability=PMHT_CLUTTER_PROBABILITY,
        modulation_match_probability=PMHT_MODULATION_MATCH_PROBABILITY,
        missed_pulse_probability=PMHT_MISSED_PULSE_PROBABILITY,
        max_pri_multiple=PMHT_MAX_PRI_MULTIPLE,
    )


def generate_pdws(scenario):
    source = SimulatedSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        emitters=scenario.emitters,
        noise_std=scenario.noise_std,
    )
    iq, metadata = source.read()
    detector = PulseDetector(
        threshold=DETECTION_THRESHOLD,
        sample_rate_hz=metadata["sample_rate_hz"],
        min_pulse_width_s=MIN_PULSE_WIDTH_S,
    )
    extractor = PDWExtractor(
        sample_rate_hz=metadata["sample_rate_hz"],
        center_frequency_hz=metadata["center_frequency_hz"],
    )
    pulses = detector.detect(iq)
    return [extractor.extract(iq, pulse) for pulse in pulses], metadata


def assignment(best_hypothesis, pdws):
    result = {pdw.pdw_id: "CLUTTER" for pdw in pdws}
    for candidate in best_hypothesis.get("candidates", []):
        for pdw in candidate["pdws"]:
            result[pdw.pdw_id] = candidate["candidate_id"]
    return result


def pairwise_partition_agreement(reference, test, pdw_ids):
    """Label-invariant agreement: do both algorithms make the same same/different decision?"""
    if len(pdw_ids) < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(len(pdw_ids)):
        for j in range(i + 1, len(pdw_ids)):
            a = pdw_ids[i]
            b = pdw_ids[j]
            ref_same = reference[a] == reference[b]
            test_same = test[a] == test[b]
            agree += int(ref_same == test_same)
            total += 1
    return agree / total if total else 1.0


def timed_associate(mht, pdws):
    start = time.perf_counter()
    hypotheses = mht.associate(pdws)
    return hypotheses, time.perf_counter() - start


def run_scenario(scenario):
    pdws, metadata = generate_pdws(scenario)

    reference_h, reference_time = timed_associate(
        make_mht(ProbabilisticMultipleHypothesisAssociator), pdws
    )
    gated_h, gated_time = timed_associate(
        make_mht(GatedFastProbabilisticMultipleHypothesisAssociator), pdws
    )

    scorer = SimulationTruthScorer(sample_rate_hz=metadata["sample_rate_hz"])
    reference_score = scorer.score(scenario, pdws, reference_h)
    gated_score = scorer.score(scenario, pdws, gated_h)

    ref_assignment = assignment(reference_h[0], pdws)
    gated_assignment = assignment(gated_h[0], pdws)
    pdw_ids = [pdw.pdw_id for pdw in pdws]
    partition_agreement = pairwise_partition_agreement(
        ref_assignment, gated_assignment, pdw_ids
    )

    ref_tracks = len(reference_h[0]["candidates"])
    gated_tracks = len(gated_h[0]["candidates"])
    truth_delta = gated_score["association_accuracy"] - reference_score["association_accuracy"]

    passed = (
        partition_agreement >= 0.99
        and gated_score["association_accuracy"] + 1e-12
        >= reference_score["association_accuracy"] - 0.01
    )

    return {
        "name": scenario.name,
        "pdws": len(pdws),
        "ref_tracks": ref_tracks,
        "gated_tracks": gated_tracks,
        "partition_agreement": partition_agreement,
        "ref_truth": reference_score["association_accuracy"],
        "gated_truth": gated_score["association_accuracy"],
        "truth_delta": truth_delta,
        "ref_time": reference_time,
        "gated_time": gated_time,
        "speedup": reference_time / gated_time if gated_time > 0 else float("inf"),
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regression-test gated-fast PMHT association against reference PMHT"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Scenario to test. May be repeated. Default: all scenarios.",
    )
    args = parser.parse_args()

    if args.scenario:
        scenarios = [get_scenario(name) for name in args.scenario]
    else:
        scenarios = list_scenarios()

    print("S2B ASSOCIATION REGRESSION")
    print("==========================")
    print("Reference PMHT vs gated-fast PMHT")
    print("Pair agreement is label-invariant and compares the best-hypothesis partition.")
    print()

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)
        print(f"{result['name']}")
        print(f"  PDWs                    : {result['pdws']}")
        print(f"  Tracks ref / gated      : {result['ref_tracks']} / {result['gated_tracks']}")
        print(f"  Partition agreement     : {100.0 * result['partition_agreement']:.1f}%")
        print(
            f"  Truth association       : "
            f"ref={100.0 * result['ref_truth']:.1f}%  "
            f"gated={100.0 * result['gated_truth']:.1f}%"
        )
        print(f"  Runtime ref / gated     : {result['ref_time']:.4f}s / {result['gated_time']:.4f}s")
        print(f"  Scenario result         : {'PASS' if result['passed'] else 'FAIL'}")
        print()

    passed = sum(result["passed"] for result in results)
    print("SUMMARY")
    print("-------")
    print(f"Scenarios passed        : {passed} / {len(results)}")
    if passed == len(results):
        print("Association regression  : PASS")
    else:
        print("Association regression  : FAIL - inspect failed scenarios before using gated-fast in main.py")


if __name__ == "__main__":
    main()
