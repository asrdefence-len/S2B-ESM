"""End-to-end diagnostic for the unified 40 MS/s S2B processing chain.

This does NOT change production behaviour. It runs sampled IQ through the same
pulse detector, PDW extractor/classifier and streaming tracker, then compares
measured PDW structure with the simulator's known source definitions. Scenario
truth is used only in this diagnostic report, never for association.
"""

from collections import Counter
import numpy as np

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource
from streaming_emitter_tracker import StreamingEmitterTracker

CENTER_HZ = 9_400_000_000.0
BLOCK_SAMPLES = 40_000
RUN_S = 20.0


def robust_interval_summary(pdws, max_gap_s=0.010):
    if len(pdws) < 2:
        return "-"
    toas = np.asarray(sorted(p.toa_s for p in pdws), dtype=float)
    dt = np.diff(toas)
    dt = dt[(dt > 0) & (dt <= max_gap_s)]
    if len(dt) == 0:
        return "-"
    us = dt * 1e6
    rounded = np.round(us / 100.0) * 100.0
    common = Counter(rounded.astype(int)).most_common(6)
    return f"median={np.median(us):.1f} us  modes=" + ", ".join(f"{v}us:{n}" for v,n in common)


def main():
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    extractor = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)
    tracker = StreamingEmitterTracker(frequency_gate_hz=2_000_000.0)

    legacy = source.legacy_scenario.emitters
    truth = []
    for e in legacy:
        truth.append((e["name"], CENTER_HZ + float(e["if_frequency_hz"]), float(e["pri_s"]), e.get("modulation", "CW")))
    truth.append(("Scripted E3", 9_410_000_000.0, 1.0e-3, "CW"))

    print("S2B END-TO-END PROCESSING CHAIN DIAGNOSTIC")
    print("==========================================")
    print(f"Sample rate       : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Run time          : {RUN_S:.1f} s")
    print(f"Tracker RF gate   : +/- {tracker.frequency_gate_hz/1e3:.0f} kHz")
    print("\nSimulator source definitions (diagnostic truth only)")
    for name, f, pri, mod in truth:
        print(f"  {name:14s} RF={f/1e6:10.3f} MHz  PRI={pri*1e6:7.1f} us  MOD={mod}")

    # Flag source separations relative to the production tracker's RF-only gate.
    print("\nSource separation versus current RF-only association gate")
    for i in range(len(truth)):
        for j in range(i+1, len(truth)):
            df = abs(truth[i][1] - truth[j][1])
            if df <= tracker.frequency_gate_hz:
                print(f"  WARNING: {truth[i][0]} <-> {truth[j][0]} separation {df/1e3:.1f} kHz is INSIDE gate")

    blocks = int(round(RUN_S * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    all_pdws = []
    for _ in range(blocks):
        iq, md = source.read_block()
        for pulse in detector.detect(iq):
            all_pdws.append(extractor.extract(iq, pulse, block_start_time_s=md["start_time_s"]))

    print(f"\nMeasured PDWs     : {len(all_pdws)}")

    # Diagnostic truth-frequency clusters use narrow windows solely to inspect the
    # measured stream before the persistent track association stage.
    print("\nMeasured PDWs near each source centre BEFORE tracker association")
    for name, f, _pri, _mod in truth:
        gate = 35_000.0 if name != "Scripted E3" else 250_000.0
        pts = [p for p in all_pdws if abs(p.frequency_hz - f) <= gate]
        mods = Counter(p.modulation_type for p in pts).most_common(4)
        amps = [p.amplitude_dbfs for p in pts]
        pws = [p.pulse_width_s*1e6 for p in pts]
        print(f"  {name:14s} N={len(pts):6d}  {robust_interval_summary(pts)}")
        if pts:
            print(f"                 PWmed={np.median(pws):.2f} us  AMPpeak={max(amps):.2f} dBFS  MOD={mods}")

    tracker.update(all_pdws)
    print("\nProduction tracker result")
    for t in tracker.tracks:
        s = t.summary()
        freqs = np.asarray([p.frequency_hz for p in t.pdws])
        lo = np.min(freqs)/1e6 if len(freqs) else 0
        hi = np.max(freqs)/1e6 if len(freqs) else 0
        print(f"  {t.emitter_id}: pulses={t.total_pulses:6d} centroid={t.frequency_hz/1e6:.3f} MHz range={lo:.3f}..{hi:.3f} MHz")
        print(f"      monitor PRI={s['pri_s']*1e6 if s['pri_s'] else 0:.1f} us  MOD={s['modulation']}  {robust_interval_summary(list(t.pdws))}")

    print("\nDIAGNOSTIC INTERPRETATION")
    print("  If two simulator sources lie inside the current +/-2 MHz RF-only gate, the")
    print("  lightweight tracker is expected to merge them. Mixed 500/1000/etc PRI blobs")
    print("  can therefore be an association artifact even when IQ detection and individual")
    print("  PDW extraction are working correctly. This diagnostic separates raw measured")
    print("  PDWs from the later tracker association so we can identify which stage is wrong.")


if __name__ == "__main__":
    main()
