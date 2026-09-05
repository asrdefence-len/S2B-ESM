import math


class PhysicalEmitterCorrelator:
    """Form hypotheses about whether pulse-sequence tracks share one physical emitter.

    This layer deliberately operates *above* the PMHT. The PMHT is allowed to
    form distinct pulse-sequence tracks when waveform/timing statistics change.
    This correlator then asks whether two such tracks may nevertheless be
    successive observable states of the same physical radar.

    The current support score is an engineering score, not a calibrated
    posterior probability. AOA/geolocation are intentionally absent for now.
    """

    def __init__(
        self,
        frequency_scale_hz=250_000.0,
        amplitude_scale_db=3.0,
        temporal_scale_s=2.0e-3,
        change_alignment_scale_s=300e-6,
    ):
        self.frequency_scale_hz = frequency_scale_hz
        self.amplitude_scale_db = amplitude_scale_db
        self.temporal_scale_s = temporal_scale_s
        self.change_alignment_scale_s = change_alignment_scale_s

    @staticmethod
    def _track_extent(candidate):
        pdws = sorted(candidate["pdws"], key=lambda pdw: pdw.toa_s)
        return pdws[0].toa_s, pdws[-1].toa_s

    @staticmethod
    def _bounded_exp(value):
        return math.exp(-max(value, 0.0))

    def _nearest_change(self, boundary_s, changes):
        if not changes:
            return None, 0.0

        nearest = min(changes, key=lambda item: abs(item["change_toa_s"] - boundary_s))
        error_s = abs(nearest["change_toa_s"] - boundary_s)
        support = self._bounded_exp(error_s / self.change_alignment_scale_s)
        return nearest, support

    def _pair_hypothesis(self, first, second, changes):
        first_start, first_end = self._track_extent(first)
        second_start, second_end = self._track_extent(second)

        # Order the tracks in time for interpretation.
        if second_start < first_start:
            first, second = second, first
            first_start, first_end, second_start, second_end = (
                second_start,
                second_end,
                first_start,
                first_end,
            )

        overlap_s = max(0.0, min(first_end, second_end) - max(first_start, second_start))
        gap_s = second_start - first_end

        frequency_delta_hz = abs(
            second["mean_frequency_hz"] - first["mean_frequency_hz"]
        )
        amplitude_delta_db = abs(
            second["mean_amplitude_dbfs"] - first["mean_amplitude_dbfs"]
        )

        frequency_support = self._bounded_exp(
            frequency_delta_hz / self.frequency_scale_hz
        )
        amplitude_support = self._bounded_exp(
            amplitude_delta_db / self.amplitude_scale_db
        )

        if overlap_s > 0.0:
            temporal_support = self._bounded_exp(
                3.0 + overlap_s / max(self.temporal_scale_s, 1e-12)
            )
        else:
            temporal_support = self._bounded_exp(
                max(gap_s, 0.0) / self.temporal_scale_s
            )

        change, change_support = self._nearest_change(second_start, changes)

        # Weighted evidence fusion. These weights are intentionally explicit so
        # they can later be replaced by calibrated likelihoods.
        support = (
            0.30 * frequency_support
            + 0.20 * amplitude_support
            + 0.30 * temporal_support
            + 0.20 * change_support
        )

        if support >= 0.80:
            assessment = "STRONG"
        elif support >= 0.60:
            assessment = "MODERATE"
        else:
            assessment = "WEAK"

        evidence = []
        if overlap_s == 0.0:
            evidence.append(f"no temporal overlap; gap={max(gap_s, 0.0) * 1e6:.1f} us")
        else:
            evidence.append(f"tracks overlap by {overlap_s * 1e6:.1f} us")

        evidence.append(f"RF separation={frequency_delta_hz / 1e3:.1f} kHz")
        evidence.append(f"level separation={amplitude_delta_db:.2f} dB")

        if change is not None:
            evidence.append(
                f"change point near T{second['candidate_id']} start "
                f"(error={abs(change['change_toa_s'] - second_start) * 1e6:.1f} us)"
            )
        else:
            evidence.append("no supporting change point")

        return {
            "track_a": first["candidate_id"],
            "track_b": second["candidate_id"],
            "support": support,
            "assessment": assessment,
            "gap_s": gap_s,
            "overlap_s": overlap_s,
            "frequency_delta_hz": frequency_delta_hz,
            "amplitude_delta_db": amplitude_delta_db,
            "change_support": change_support,
            "evidence": evidence,
        }

    def correlate(self, best_hypothesis, changes):
        candidates = best_hypothesis.get("candidates", [])
        hypotheses = []

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                hypotheses.append(
                    self._pair_hypothesis(candidates[i], candidates[j], changes)
                )

        return sorted(hypotheses, key=lambda item: item["support"], reverse=True)


def print_physical_emitter_hypotheses(hypotheses):
    print("PHYSICAL EMITTER CORRELATION")
    print("============================")
    print("Engineering hypothesis layer: asks whether PMHT sequence tracks may share one physical emitter.")
    print("Support values are engineered evidence scores, not calibrated probabilities.")
    print()

    if not hypotheses:
        print("No pairwise track-correlation hypothesis available.")
        print()
        return

    for index, hypothesis in enumerate(hypotheses, start=1):
        print(
            f"H{index}: T{hypothesis['track_a']} + T{hypothesis['track_b']} "
            f"same physical emitter: {hypothesis['assessment']} "
            f"support={100.0 * hypothesis['support']:.1f}%"
        )
        for evidence in hypothesis["evidence"]:
            print(f"  - {evidence}")
        print()
