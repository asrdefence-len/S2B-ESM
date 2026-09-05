class OperatorEmitterSummary:
    """Build a concise operator-facing emitter picture from the best MHT hypothesis.

    This deliberately hides most PDW-level detail. The operator sees inferred
    emitters and their current measured characteristics, while lower-level PDW
    and hypothesis diagnostics remain available elsewhere for analysis.
    """

    def build(self, hypotheses, family_marginals, track_membership):
        if not hypotheses:
            return []

        best = hypotheses[0]
        summaries = []

        for candidate in best["candidates"]:
            candidate_id = candidate["candidate_id"]
            pdws = candidate["pdws"]

            family_weights = []
            track_weights = []
            for pdw in pdws:
                family_distribution = family_marginals.get(pdw.pdw_id, {})
                family_weights.append(family_distribution.get(candidate_id, 0.0))
                track_weights.append(
                    track_membership.get(pdw.pdw_id, {}).get(candidate_id, 0.0)
                )

            family_confidence = (
                sum(family_weights) / len(family_weights)
                if family_weights else 0.0
            )
            track_confidence = (
                sum(track_weights) / len(track_weights)
                if track_weights else 0.0
            )

            pri_s = candidate.get("estimated_pri_s")

            summaries.append(
                {
                    "emitter_id": candidate_id,
                    "pulse_count": len(pdws),
                    "frequency_hz": candidate["mean_frequency_hz"],
                    "pulse_width_s": candidate["mean_pulse_width_s"],
                    "amplitude_dbfs": candidate["mean_amplitude_dbfs"],
                    "modulation": candidate["dominant_modulation"],
                    "pri_s": pri_s,
                    "family_confidence": family_confidence,
                    "track_confidence": track_confidence,
                }
            )

        return summaries


def _confidence_text(value):
    if value >= 0.95:
        return "HIGH"
    if value >= 0.75:
        return "MED"
    return "LOW"


def print_operator_picture(summaries):
    print("ESM OPERATOR PICTURE")
    print("====================")
    print(f"Emitters detected: {len(summaries)}")
    print()

    for emitter in summaries:
        pri_s = emitter["pri_s"]
        pri_text = "UNRESOLVED" if pri_s is None else f"{pri_s * 1e6:.1f} us"

        print(f"EMITTER {emitter['emitter_id']}")
        print(f"  RF          : {emitter['frequency_hz'] / 1e6:.3f} MHz")
        print(f"  PRI         : {pri_text}")
        print(f"  PW          : {emitter['pulse_width_s'] * 1e6:.3f} us")
        print(f"  MODULATION  : {emitter['modulation']}")
        print(f"  LEVEL       : {emitter['amplitude_dbfs']:.2f} dBFS")
        print(f"  PULSES      : {emitter['pulse_count']}")
        print(
            f"  EMITTER ID  : {_confidence_text(emitter['family_confidence'])} "
            f"({100.0 * emitter['family_confidence']:.1f}%)"
        )
        print(
            f"  TRACK       : {_confidence_text(emitter['track_confidence'])} "
            f"({100.0 * emitter['track_confidence']:.1f}%)"
        )
        print()
