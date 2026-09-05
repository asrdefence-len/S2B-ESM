import statistics


class FrequencyAssociator:
    """Simple baseline PDW association using frequency only."""

    def __init__(self, frequency_tolerance_hz):
        self.frequency_tolerance_hz = frequency_tolerance_hz

    def associate(self, pdws):
        groups = []

        for pdw in pdws:
            best_group = None
            best_error_hz = None

            for group in groups:
                error_hz = abs(pdw.frequency_hz - group["mean_frequency_hz"])
                if error_hz <= self.frequency_tolerance_hz:
                    if best_error_hz is None or error_hz < best_error_hz:
                        best_group = group
                        best_error_hz = error_hz

            if best_group is None:
                groups.append({
                    "candidate_id": len(groups) + 1,
                    "mean_frequency_hz": pdw.frequency_hz,
                    "pdws": [pdw],
                })
                continue

            best_group["pdws"].append(pdw)
            best_group["mean_frequency_hz"] = sum(
                item.frequency_hz for item in best_group["pdws"]
            ) / len(best_group["pdws"])

        return groups


class EvidenceAssociator:
    """Greedy multi-feature association using measured PDW evidence only."""

    def __init__(
        self,
        frequency_tolerance_hz,
        pulse_width_tolerance_s,
        timing_tolerance_s,
        amplitude_tolerance_db,
        max_pri_multiple=3,
    ):
        self.frequency_tolerance_hz = frequency_tolerance_hz
        self.pulse_width_tolerance_s = pulse_width_tolerance_s
        self.timing_tolerance_s = timing_tolerance_s
        self.amplitude_tolerance_db = amplitude_tolerance_db
        self.max_pri_multiple = max_pri_multiple

    def _update_group(self, group):
        pdws = group["pdws"]
        group["mean_frequency_hz"] = sum(p.frequency_hz for p in pdws) / len(pdws)
        group["mean_pulse_width_s"] = sum(p.pulse_width_s for p in pdws) / len(pdws)
        group["mean_amplitude_dbfs"] = sum(p.amplitude_dbfs for p in pdws) / len(pdws)

        modulation_counts = {}
        for p in pdws:
            modulation_counts[p.modulation_type] = modulation_counts.get(p.modulation_type, 0) + 1
        group["dominant_modulation"] = max(modulation_counts, key=modulation_counts.get)

        if len(pdws) >= 2:
            intervals = [pdws[i].toa_s - pdws[i - 1].toa_s for i in range(1, len(pdws))]
            group["estimated_pri_s"] = statistics.median(intervals)
        else:
            group["estimated_pri_s"] = None

    def _timing_error(self, pdw, group):
        estimated_pri_s = group["estimated_pri_s"]
        if estimated_pri_s is None:
            return None

        dt_s = pdw.toa_s - group["pdws"][-1].toa_s
        if dt_s <= 0:
            return float("inf")

        return min(
            abs(dt_s - multiple * estimated_pri_s)
            for multiple in range(1, self.max_pri_multiple + 1)
        )

    def _score(self, pdw, group):
        frequency_error_hz = abs(pdw.frequency_hz - group["mean_frequency_hz"])
        pulse_width_error_s = abs(pdw.pulse_width_s - group["mean_pulse_width_s"])
        amplitude_error_db = abs(pdw.amplitude_dbfs - group["mean_amplitude_dbfs"])

        if frequency_error_hz > self.frequency_tolerance_hz:
            return None
        if pulse_width_error_s > self.pulse_width_tolerance_s:
            return None
        if amplitude_error_db > self.amplitude_tolerance_db:
            return None

        score = (
            frequency_error_hz / self.frequency_tolerance_hz
            + pulse_width_error_s / self.pulse_width_tolerance_s
            + 0.5 * amplitude_error_db / self.amplitude_tolerance_db
        )

        # Modulation is strong supporting evidence, but not an absolute gate.
        if pdw.modulation_type != group["dominant_modulation"]:
            score += 1.0

        timing_error_s = self._timing_error(pdw, group)
        if timing_error_s is not None:
            if timing_error_s > self.timing_tolerance_s:
                return None
            score += timing_error_s / self.timing_tolerance_s

        return score

    def associate(self, pdws):
        groups = []

        for pdw in pdws:
            best_group = None
            best_score = None

            for group in groups:
                score = self._score(pdw, group)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_group = group
                    best_score = score

            if best_group is None:
                new_group = {
                    "candidate_id": len(groups) + 1,
                    "pdws": [pdw],
                    "mean_frequency_hz": pdw.frequency_hz,
                    "mean_pulse_width_s": pdw.pulse_width_s,
                    "mean_amplitude_dbfs": pdw.amplitude_dbfs,
                    "dominant_modulation": pdw.modulation_type,
                    "estimated_pri_s": None,
                }
                groups.append(new_group)
                continue

            best_group["pdws"].append(pdw)
            self._update_group(best_group)

        return groups
