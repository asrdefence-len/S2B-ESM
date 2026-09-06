from dataclasses import dataclass


@dataclass(frozen=True)
class EmitterLibraryMatch:
    emitter_type: str
    confidence: float
    reason: str


class EmitterLibrary:
    """Small observable-feature emitter library.

    This deliberately uses only ESM-derived/observable features. It must not
    inspect scripted radar truth such as the scenario's hidden mode name.
    """

    def identify(self, frequency_hz, pri_s, waveform, illumination_state=None):
        if frequency_hz is None or pri_s is None:
            return EmitterLibraryMatch("UNKNOWN", 0.0, "Insufficient observable parameters")

        rf_ok = abs(float(frequency_hz) - 9.410e9) <= 25.0e6
        pri_ok = 0.0008 <= float(pri_s) <= 0.0012
        waveform_ok = str(waveform).upper() == "CW"
        rotating_ok = str(illumination_state).upper() == "PERIODIC_SCAN"

        if rf_ok and pri_ok and waveform_ok and rotating_ok:
            return EmitterLibraryMatch(
                "NAVRADAR", 0.90,
                "9.41 GHz class + ~1 ms PRI + CW + periodic rotating illumination",
            )

        return EmitterLibraryMatch("UNKNOWN", 0.0, "No library entry matched all observable features")
