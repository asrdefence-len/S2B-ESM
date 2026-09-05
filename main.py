from config import *

from simulated_source import SimulatedSource
from pulse_detector import PulseDetector
from pdw_extractor import PDWExtractor
from pulse_sequence import PulseSequenceAnalyzer
from association import FrequencyAssociator, EvidenceAssociator
from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator
from change_detection import PDWChangeDetector, print_change_detections
from physical_emitter_correlation import PhysicalEmitterCorrelator, print_physical_emitter_hypotheses
from operator_display import OperatorEmitterSummary, print_operator_picture
from scenario_runner import select_scenario
from truth_scoring import SimulationTruthScorer, print_truth_score


def print_groups(title, groups, analyzer):
    print(title)
    print("-" * len(title))

    for group in groups:
        sequence = analyzer.analyze(group["pdws"])
        candidate_id = group["candidate_id"]
        mean_frequency_mhz = group["mean_frequency_hz"] / 1e6
        extra = ""

        if "mean_pulse_width_s" in group:
            mean_pw_us = group["mean_pulse_width_s"] * 1e6
            mean_amp_dbfs = group["mean_amplitude_dbfs"]
            modulation = group["dominant_modulation"]
            estimated_pri_s = group.get("estimated_pri_s")
            pri_text = "unknown" if estimated_pri_s is None else f"{estimated_pri_s * 1e6:.3f} us"
            extra = (
                f", mean PW={mean_pw_us:.3f} us"
                f", mean AMP={mean_amp_dbfs:.2f} dBFS"
                f", modulation={modulation}"
                f", estimated PRI={pri_text}"
            )

        print(
            f"Candidate {candidate_id}: {len(group['pdws'])} pulses, "
            f"mean frequency={mean_frequency_mhz:.3f} MHz{extra}"
        )

        for item in sequence:
            pdw = item["pdw"]
            pri_s = item["pri_s"]
            pri_text = "--------" if pri_s is None else f"{pri_s * 1e6:8.3f} us"
            print(
                f"  PDW {pdw.pdw_id:06d}  "
                f"PW={pdw.pulse_width_s * 1e6:6.3f} us  "
                f"AMP={pdw.amplitude_dbfs:6.2f} dBFS  "
                f"MOD={pdw.modulation_type:3s}  "
                f"BW={pdw.modulation_bandwidth_hz / 1e6:5.3f} MHz  "
                f"PRI={pri_text}"
            )
        print()


def print_hypotheses(hypotheses):
    print("Probabilistic multiple-hypothesis association")
    print("---------------------------------------------")

    for rank, hypothesis in enumerate(hypotheses[:MHT_DISPLAY_HYPOTHESES], start=1):
        clutter_count = len(hypothesis.get("clutter_pdws", []))
        print(
            f"Hypothesis {rank}: posterior={100.0 * hypothesis['probability']:.2f}%  "
            f"neg-log-weight={hypothesis['score']:.3f}  "
            f"tracks={len(hypothesis['candidates'])}  clutter={clutter_count}"
        )
        for candidate in hypothesis["candidates"]:
            pri_s = candidate.get("estimated_pri_s")
            pri_text = "unknown" if pri_s is None else f"{pri_s * 1e6:.1f} us"
            pdw_ids = ",".join(str(p.pdw_id) for p in candidate["pdws"])
            print(
                f"  Track {candidate['candidate_id']}: {len(candidate['pdws'])} pulses, "
                f"F={candidate['mean_frequency_hz'] / 1e6:.3f} MHz, "
                f"PW={candidate['mean_pulse_width_s'] * 1e6:.3f} us, "
                f"AMP={candidate['mean_amplitude_dbfs']:.2f} dBFS, "
                f"MOD={candidate['dominant_modulation']}, PRI={pri_text}"
            )
            print(f"    PDWs: {pdw_ids}")
        if clutter_count:
            clutter_ids = ",".join(str(p.pdw_id) for p in hypothesis["clutter_pdws"])
            print(f"  Clutter PDWs: {clutter_ids}")
        print()


def print_association_uncertainty(pdws, marginals, track_membership, best_hypothesis):
    print("Per-PDW association uncertainty")
    print("-------------------------------")
    print("Family = sequence-family posterior mass; Track = co-association posterior mass.")

    best_assignment = {}
    for candidate in best_hypothesis["candidates"]:
        for pdw in candidate["pdws"]:
            best_assignment[pdw.pdw_id] = candidate["candidate_id"]

    for pdw in pdws:
        candidate_id = best_assignment.get(pdw.pdw_id)
        if candidate_id is None:
            other_weight = marginals.get(pdw.pdw_id, {}).get("OTHER", 0.0)
            print(
                f"PDW {pdw.pdw_id:06d}  CLUTTER  "
                f"posterior={100.0 * other_weight:5.1f}%"
            )
            continue

        family_weight = marginals[pdw.pdw_id].get(candidate_id, 0.0)
        track_weight = track_membership[pdw.pdw_id].get(candidate_id, 0.0)

        if track_weight >= 0.95:
            confidence = "HIGH"
        elif track_weight >= 0.75:
            confidence = "MED "
        else:
            confidence = "LOW "

        print(
            f"PDW {pdw.pdw_id:06d}  T{candidate_id}  "
            f"family={100.0 * family_weight:5.1f}%  "
            f"track={100.0 * track_weight:5.1f}%  "
            f"track confidence={confidence}"
        )
    print()


def main():
    scenario = select_scenario()
    if scenario is None:
        return

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
    pdws = [extractor.extract(iq, pulse) for pulse in pulses]
    analyzer = PulseSequenceAnalyzer()
    raw_sequence = analyzer.analyze(pdws)

    change_detector = PDWChangeDetector(
        frequency_scale_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
        pulse_width_scale_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,
        amplitude_scale_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
        pri_scale_s=ASSOCIATION_TIMING_TOLERANCE_S,
        bandwidth_scale_hz=MODULATION_LFM_BANDWIDTH_THRESHOLD_HZ,
    )
    changes = change_detector.detect(pdws)

    frequency_groups = FrequencyAssociator(
        frequency_tolerance_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ
    ).associate(pdws)

    evidence_groups = EvidenceAssociator(
        frequency_tolerance_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
        pulse_width_tolerance_s=ASSOCIATION_PULSE_WIDTH_TOLERANCE_S,
        timing_tolerance_s=ASSOCIATION_TIMING_TOLERANCE_S,
        amplitude_tolerance_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
    ).associate(pdws)

    mht = ProbabilisticMultipleHypothesisAssociator(
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
    hypotheses = mht.associate(pdws)
    marginals = mht.association_marginals(hypotheses)
    track_membership = mht.reference_track_membership(hypotheses)

    physical_hypotheses = PhysicalEmitterCorrelator(
        frequency_scale_hz=ASSOCIATION_FREQUENCY_TOLERANCE_HZ,
        amplitude_scale_db=ASSOCIATION_AMPLITUDE_TOLERANCE_DB,
    ).correlate(hypotheses[0], changes)

    operator_summaries = OperatorEmitterSummary().build(
        hypotheses,
        marginals,
        track_membership,
    )

    truth_score = SimulationTruthScorer(
        sample_rate_hz=metadata["sample_rate_hz"]
    ).score(scenario, pdws, hypotheses)

    print("ENGINEERING / ANALYSIS DETAIL")
    print("=============================")
    print()
    print("S2B Experimental ESM")
    print("--------------------")
    print(f"Scenario            : {scenario.name}")
    print(f"Description         : {scenario.description}")
    print(f"Configured segments : {len(scenario.emitters)}")
    print(f"Pulses detected     : {len(pulses)}")
    print()

    print("Raw interleaved PDW stream")
    print("--------------------------")
    for item in raw_sequence:
        pdw = item["pdw"]
        pri_s = item["pri_s"]
        pri_text = "--------" if pri_s is None else f"{pri_s * 1e6:8.3f} us"
        print(f"{pdw}  PRI={pri_text}")

    print()
    print_change_detections(changes)
    print_groups("Baseline candidate groups - frequency association only", frequency_groups, analyzer)
    print_groups(
        "Greedy evidence groups - frequency + PW + amplitude + modulation + PRI",
        evidence_groups,
        analyzer,
    )
    print_hypotheses(hypotheses)
    print_association_uncertainty(pdws, marginals, track_membership, hypotheses[0])
    print_physical_emitter_hypotheses(physical_hypotheses)

    print_truth_score(truth_score)
    print_operator_picture(operator_summaries, physical_hypotheses)


if __name__ == "__main__":
    main()
