import statistics


class OperatorEmitterSummary:
    """Build concise operator-facing summaries of PMHT pulse-sequence tracks."""

    @staticmethod
    def _pri_summary(pdws):
        if len(pdws) < 2:
            return {
                "pri_s": None,
                "pri_min_s": None,
                "pri_max_s": None,
                "pri_std_s": None,
                "pri_pattern": "UNRESOLVED",
            }

        intervals = [
            pdws[index].toa_s - pdws[index - 1].toa_s
            for index in range(1, len(pdws))
        ]
        pri_s = statistics.median(intervals)
        pri_min_s = min(intervals)
        pri_max_s = max(intervals)
        pri_std_s = statistics.pstdev(intervals) if len(intervals) > 1 else 0.0

        relative_std = pri_std_s / pri_s if pri_s > 0.0 else 0.0
        pri_pattern = "JITTERED" if relative_std > 0.005 else "STABLE"

        return {
            "pri_s": pri_s,
            "pri_min_s": pri_min_s,
            "pri_max_s": pri_max_s,
            "pri_std_s": pri_std_s,
            "pri_pattern": pri_pattern,
        }

    def build(self, hypotheses, family_marginals, track_membership):
        if not hypotheses:
            return []

        best = hypotheses[0]
        summaries = []

        for candidate in best["candidates"]:
            candidate_id = candidate["candidate_id"]
            pdws = sorted(candidate["pdws"], key=lambda pdw: pdw.toa_s)

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
            mean_aoa_deg = (
                sum(getattr(pdw, "aoa_deg", 0.0) for pdw in pdws) / len(pdws)
                if pdws else 0.0
            )

            summaries.append(
                {
                    "track_id": candidate_id,
                    "pulse_count": len(pdws),
                    "start_toa_s": pdws[0].toa_s,
                    "end_toa_s": pdws[-1].toa_s,
                    "frequency_hz": candidate["mean_frequency_hz"],
                    "aoa_deg": mean_aoa_deg,
                    "pulse_width_s": candidate["mean_pulse_width_s"],
                    "amplitude_dbfs": candidate["mean_amplitude_dbfs"],
                    "modulation": candidate["dominant_modulation"],
                    **self._pri_summary(pdws),
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


def _physical_groups(summaries, physical_hypotheses, strong_threshold=0.80):
    """Group sequence tracks using only STRONG physical-correlation hypotheses."""
    track_ids = [summary["track_id"] for summary in summaries]
    parent = {track_id: track_id for track_id in track_ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    accepted = []
    for hypothesis in physical_hypotheses or []:
        if hypothesis["support"] >= strong_threshold:
            union(hypothesis["track_a"], hypothesis["track_b"])
            accepted.append(hypothesis)

    groups = {}
    for summary in summaries:
        groups.setdefault(find(summary["track_id"]), []).append(summary)

    result = []
    for tracks in groups.values():
        tracks = sorted(tracks, key=lambda item: item["start_toa_s"])
        ids = {track["track_id"] for track in tracks}
        links = [
            hypothesis
            for hypothesis in accepted
            if hypothesis["track_a"] in ids and hypothesis["track_b"] in ids
        ]
        result.append({"tracks": tracks, "links": links})

    return sorted(result, key=lambda group: group["tracks"][0]["start_toa_s"])


def _format_pri(track):
    if track["pri_s"] is None:
        return "UNRESOLVED"
    return f"{track['pri_s'] * 1e6:.1f} us"


def _print_track(track, indent="  "):
    print(f"{indent}SEQUENCE TRACK T{track['track_id']}")
    print(f"{indent}  RF          : {track['frequency_hz'] / 1e6:.3f} MHz")
    print(f"{indent}  AOA         : {track.get('aoa_deg', 0.0):.1f} deg")
    print(f"{indent}  PRI MEDIAN  : {_format_pri(track)}")
    print(f"{indent}  PRI PATTERN : {track['pri_pattern']}")
    print(f"{indent}  PW          : {track['pulse_width_s'] * 1e6:.3f} us")
    print(f"{indent}  MODULATION  : {track['modulation']}")
    print(f"{indent}  LEVEL       : {track['amplitude_dbfs']:.2f} dBFS")
    print(f"{indent}  PULSES      : {track['pulse_count']}")
    print(
        f"{indent}  SEQ FAMILY  : {_confidence_text(track['family_confidence'])} "
        f"({100.0 * track['family_confidence']:.1f}%)"
    )
    print(
        f"{indent}  SEQ TRACK   : {_confidence_text(track['track_confidence'])} "
        f"({100.0 * track['track_confidence']:.1f}%)"
    )


def print_operator_picture(summaries, physical_hypotheses=None):
    groups = _physical_groups(summaries, physical_hypotheses)

    print("ESM OPERATOR PICTURE")
    print("====================")
    print(f"Physical emitters assessed : {len(groups)}")
    print(f"Pulse-sequence tracks      : {len(summaries)}")
    print()

    for emitter_index, group in enumerate(groups, start=1):
        tracks = group["tracks"]
        links = group["links"]
        current = max(tracks, key=lambda item: item["end_toa_s"])

        print(f"PHYSICAL EMITTER E{emitter_index}")

        if len(tracks) > 1:
            strongest = max(links, key=lambda item: item["support"]) if links else None
            print("  OBSERVABLE STATE CHANGE : DETECTED")
            if strongest is not None:
                print(
                    f"  TRACK CORRELATION        : {strongest['assessment']} "
                    f"({100.0 * strongest['support']:.1f}% evidence support; not probability)"
                )
            print(f"  LINKED SEQUENCE TRACKS   : {', '.join('T' + str(t['track_id']) for t in tracks)}")
            print()
            print("  CURRENT OBSERVED STATE")
            _print_track(current, indent="    ")

            previous_tracks = [track for track in tracks if track["track_id"] != current["track_id"]]
            if previous_tracks:
                print()
                print("  PREVIOUS OBSERVED STATE(S)")
                for track in previous_tracks:
                    _print_track(track, indent="    ")
        else:
            print("  OBSERVABLE STATE CHANGE : none associated")
            _print_track(current, indent="  ")

        print()
