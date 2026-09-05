import statistics


class PDWChangeDetector:
    """Experimental multivariate change-point detector for an ordered PDW stream.

    The detector compares short windows immediately before and after each
    possible split. It looks for joint changes in PRI, RF, pulse width,
    amplitude, modulation and modulation bandwidth. The resulting score is an
    engineering change-evidence score, not a calibrated probability.

    Important: this module does not decide what the change means. A strong
    change can represent a mode transition, another interleaved sequence, a new
    emitter, or another change in the observation process.
    """

    def __init__(
        self,
        frequency_scale_hz=250_000.0,
        pulse_width_scale_s=1.0e-6,
        amplitude_scale_db=3.0,
        pri_scale_s=80e-6,
        bandwidth_scale_hz=200_000.0,
        modulation_change_weight=6.0,
        min_window_pdws=4,
        max_window_pdws=6,
        detection_threshold=12.0,
    ):
        self.frequency_scale_hz = frequency_scale_hz
        self.pulse_width_scale_s = pulse_width_scale_s
        self.amplitude_scale_db = amplitude_scale_db
        self.pri_scale_s = pri_scale_s
        self.bandwidth_scale_hz = bandwidth_scale_hz
        self.modulation_change_weight = modulation_change_weight
        self.min_window_pdws = min_window_pdws
        self.max_window_pdws = max_window_pdws
        self.detection_threshold = detection_threshold

    @staticmethod
    def _mean(values):
        return sum(values) / len(values)

    @staticmethod
    def _dominant_modulation(pdws):
        counts = {}
        for pdw in pdws:
            counts[pdw.modulation_type] = counts.get(pdw.modulation_type, 0) + 1
        return max(counts, key=counts.get)

    @staticmethod
    def _median_pri(pdws):
        if len(pdws) < 2:
            return None
        intervals = [
            pdws[index].toa_s - pdws[index - 1].toa_s
            for index in range(1, len(pdws))
        ]
        return statistics.median(intervals)

    def _summarize(self, pdws):
        return {
            "frequency_hz": self._mean([p.frequency_hz for p in pdws]),
            "pulse_width_s": self._mean([p.pulse_width_s for p in pdws]),
            "amplitude_dbfs": self._mean([p.amplitude_dbfs for p in pdws]),
            "bandwidth_hz": self._mean([p.modulation_bandwidth_hz for p in pdws]),
            "modulation": self._dominant_modulation(pdws),
            "pri_s": self._median_pri(pdws),
        }

    @staticmethod
    def _normalized_square(delta, scale):
        if scale <= 0.0:
            return 0.0
        return (delta / scale) ** 2

    def _score_split(self, before, after):
        left = self._summarize(before)
        right = self._summarize(after)

        components = {
            "frequency": self._normalized_square(
                right["frequency_hz"] - left["frequency_hz"],
                self.frequency_scale_hz,
            ),
            "pulse_width": self._normalized_square(
                right["pulse_width_s"] - left["pulse_width_s"],
                self.pulse_width_scale_s,
            ),
            "amplitude": self._normalized_square(
                right["amplitude_dbfs"] - left["amplitude_dbfs"],
                self.amplitude_scale_db,
            ),
            "bandwidth": self._normalized_square(
                right["bandwidth_hz"] - left["bandwidth_hz"],
                self.bandwidth_scale_hz,
            ),
        }

        if left["pri_s"] is not None and right["pri_s"] is not None:
            components["pri"] = self._normalized_square(
                right["pri_s"] - left["pri_s"],
                self.pri_scale_s,
            )
        else:
            components["pri"] = 0.0

        components["modulation"] = (
            self.modulation_change_weight
            if left["modulation"] != right["modulation"]
            else 0.0
        )

        return sum(components.values()), components, left, right

    def detect(self, pdws):
        ordered = sorted(pdws, key=lambda pdw: pdw.toa_s)
        minimum = self.min_window_pdws

        if len(ordered) < 2 * minimum:
            return []

        candidates = []

        for split_index in range(minimum, len(ordered) - minimum + 1):
            before_start = max(0, split_index - self.max_window_pdws)
            after_end = min(len(ordered), split_index + self.max_window_pdws)
            before = ordered[before_start:split_index]
            after = ordered[split_index:after_end]

            if len(before) < minimum or len(after) < minimum:
                continue

            score, components, left, right = self._score_split(before, after)
            if score < self.detection_threshold:
                continue

            first_after = ordered[split_index]
            candidates.append(
                {
                    "score": score,
                    "split_index": split_index,
                    "first_after_pdw_id": first_after.pdw_id,
                    "change_toa_s": first_after.toa_s,
                    "components": components,
                    "before": left,
                    "after": right,
                }
            )

        # Suppress nearby duplicate detections by keeping only the strongest
        # candidate within +/- min_window_pdws pulses.
        selected = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if any(
                abs(candidate["split_index"] - existing["split_index"]) < minimum
                for existing in selected
            ):
                continue
            selected.append(candidate)

        return sorted(selected, key=lambda item: item["change_toa_s"])


def print_change_detections(changes):
    print("MULTIVARIATE PDW CHANGE DETECTION")
    print("=================================")
    print("Engineering diagnostic: detects observable signal-process changes; does not assign mode meaning.")
    print()

    if not changes:
        print("No significant change point detected.")
        print()
        return

    for index, change in enumerate(changes, start=1):
        before = change["before"]
        after = change["after"]
        components = change["components"]

        print(
            f"Change {index}: before PDW {change['first_after_pdw_id']:06d}  "
            f"TOA={change['change_toa_s'] * 1e3:.3f} ms  "
            f"evidence score={change['score']:.1f}"
        )
        print(
            f"  RF          : {before['frequency_hz'] / 1e6:.3f} -> "
            f"{after['frequency_hz'] / 1e6:.3f} MHz"
        )
        if before["pri_s"] is not None and after["pri_s"] is not None:
            print(
                f"  PRI         : {before['pri_s'] * 1e6:.1f} -> "
                f"{after['pri_s'] * 1e6:.1f} us"
            )
        print(
            f"  PW          : {before['pulse_width_s'] * 1e6:.3f} -> "
            f"{after['pulse_width_s'] * 1e6:.3f} us"
        )
        print(
            f"  MODULATION  : {before['modulation']} -> {after['modulation']}"
        )
        print(
            f"  BANDWIDTH   : {before['bandwidth_hz'] / 1e6:.3f} -> "
            f"{after['bandwidth_hz'] / 1e6:.3f} MHz"
        )
        print(
            f"  LEVEL       : {before['amplitude_dbfs']:.2f} -> "
            f"{after['amplitude_dbfs']:.2f} dBFS"
        )

        ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
        evidence_text = ", ".join(
            f"{name}={value:.1f}" for name, value in ranked if value > 0.05
        )
        print(f"  Evidence    : {evidence_text}")
        print()
