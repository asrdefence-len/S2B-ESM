class FrequencyAssociator:
    """Simple first-pass PDW association using frequency only.

    This deliberately avoids simulator truth. Each PDW is assigned to the
    nearest existing candidate emitter group when its frequency lies within
    the configured tolerance; otherwise a new candidate group is created.
    """

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
                groups.append(
                    {
                        "candidate_id": len(groups) + 1,
                        "mean_frequency_hz": pdw.frequency_hz,
                        "pdws": [pdw],
                    }
                )
                continue

            best_group["pdws"].append(pdw)
            best_group["mean_frequency_hz"] = sum(
                item.frequency_hz for item in best_group["pdws"]
            ) / len(best_group["pdws"])

        return groups
