from dataclasses import dataclass


@dataclass(frozen=True)
class EmitterLibraryMatch:
    emitter_type: str
    confidence: float
    reason: str


class EmitterLibrary:
    """Small observable-feature emitter library.

    Identity and mode are deliberately separated. A radar that changes mode does
    not instantly become a different physical emitter. A strong prior library
    match may therefore be retained as a lower-confidence hypothesis when its
    current mode no longer matches the original library signature.

    Only ESM-derived/observable features are used; scripted mode truth is never
    inspected.
    """

    def identify(
        self,
        frequency_hz,
        pri_s,
        waveform,
        illumination_state=None,
        previous_type=None,
    ):
        if frequency_hz is None or pri_s is None:
            return EmitterLibraryMatch("UNKNOWN", 0.0, "Insufficient observable parameters")

        rf_ok = abs(float(frequency_hz) - 9.410e9) <= 25.0e6
        pri_ok = 0.0008 <= float(pri_s) <= 0.0012
        waveform_ok = str(waveform).upper() == "CW"
        rotating_ok = str(illumination_state).upper() == "PERIODIC_SCAN"

        if rf_ok and pri_ok and waveform_ok and rotating_ok:
            return EmitterLibraryMatch(
                "NAVRADAR",
                0.90,
                "9.41 GHz class + ~1 ms PRI + CW + periodic rotating illumination",
            )

        # Identity persistence: if this same physical emitter was previously a
        # strong NAVRADAR match and remains in the expected RF region, retain the
        # identity as a hypothesis while explicitly reducing confidence. The
        # question mark tells the operator that the present mode does not match
        # the defining library signature.
        if rf_ok and str(previous_type or "").upper() in ("NAVRADAR", "NAVRADAR?"):
            matched = sum((pri_ok, waveform_ok, rotating_ok))
            confidence = 0.45 + 0.10 * matched
            return EmitterLibraryMatch(
                "NAVRADAR?",
                min(confidence, 0.70),
                "Prior NAVRADAR identity retained; current mode no longer matches the full NAVRADAR signature",
            )

        return EmitterLibraryMatch(
            "UNKNOWN",
            0.0,
            "No library entry matched the current observable features",
        )
