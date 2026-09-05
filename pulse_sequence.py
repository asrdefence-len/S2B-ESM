class PulseSequenceAnalyzer:
    def analyze(self, pdws):
        results = []
        previous_toa = None

        for pdw in pdws:
            pri_s = None

            if previous_toa is not None:
                pri_s = pdw.toa_s - previous_toa

            results.append({
                "pdw": pdw,
                "pri_s": pri_s,
            })

            previous_toa = pdw.toa_s

        return results
