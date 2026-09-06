"""Diagnose E3 using only PDWs measured from the common 40 MS/s IQ stream.

E1, E2 and E3 are mixed into the same sampled IQ blocks.  The diagnostic then
selects the 9.410 GHz neighbourhood *after* pulse detection and PDW extraction.
No scripted emitter ID, antenna azimuth, received-power truth or mode label is
used to decide which detections are reported as the E3 candidate.
"""

from collections import defaultdict

from config import DETECTION_THRESHOLD, MIN_PULSE_WIDTH_S, SAMPLE_RATE_HZ
from pdw_extractor import PDWExtractor
from pulse_detector import PulseDetector
from simulated_streaming_source import SimulatedStreamingIQSource


CENTER_HZ = 9_400_000_000
BLOCK_SAMPLES = 40_000       # 1 ms at 40 MS/s
RUN_SECONDS = 15.0           # enough for repeated 2 s scan evidence before dwell
E3_NOMINAL_HZ = 9_410_000_000
E3_WINDOW_HZ = 2_000_000     # observation-side RF gate only


def is_e3_candidate(pdw):
    return abs(pdw.frequency_hz - E3_NOMINAL_HZ) <= E3_WINDOW_HZ


def main():
    source = SimulatedStreamingIQSource(
        sample_rate_hz=SAMPLE_RATE_HZ,
        center_frequency_hz=CENTER_HZ,
        block_samples=BLOCK_SAMPLES,
        noise_std=0.02,
    )
    detector = PulseDetector(DETECTION_THRESHOLD, SAMPLE_RATE_HZ, MIN_PULSE_WIDTH_S)
    extractor = PDWExtractor(SAMPLE_RATE_HZ, CENTER_HZ)

    blocks = int(round(RUN_SECONDS * SAMPLE_RATE_HZ / BLOCK_SAMPLES))
    all_count = 0
    e3_pdws = []
    by_second = defaultdict(list)

    print("S2B E3 MEASURED-PDW DIAGNOSTIC")
    print("===============================")
    print(f"Sample rate       : {SAMPLE_RATE_HZ/1e6:.1f} MS/s")
    print(f"Block size        : {BLOCK_SAMPLES} samples ({1000*BLOCK_SAMPLES/SAMPLE_RATE_HZ:.3f} ms)")
    print(f"Run time          : {RUN_SECONDS:.1f} s")
    print(f"Candidate RF gate : {E3_NOMINAL_HZ/1e6:.3f} +/- {E3_WINDOW_HZ/1e6:.3f} MHz")
    print("Selection point   : AFTER common IQ detector + waveform classifier + PDW extractor")
    print("Scenario truth    : NOT used for candidate selection\n")

    for _ in range(blocks):
        iq, metadata = source.read_block()
        for pulse in detector.detect(iq):
            pdw = extractor.extract(iq, pulse, block_start_time_s=metadata["start_time_s"])
            all_count += 1
            if is_e3_candidate(pdw):
                e3_pdws.append(pdw)
                by_second[int(pdw.toa_s)].append(pdw)

    print("Measured 9.410 GHz candidate activity by second")
    print("sec   PDWs   first TOA    last TOA     peak amp   mean amp   waveform(s)")
    print("---  -----  ----------  ----------  ----------  ---------  ----------------")
    for sec in range(int(RUN_SECONDS)):
        group = by_second.get(sec, [])
        if not group:
            print(f"{sec:3d}      0       -           -           -          -      -")
            continue
        amps = [p.amplitude_dbfs for p in group]
        mods = sorted(set(p.modulation_type for p in group))
        print(
            f"{sec:3d}  {len(group):5d}  {group[0].toa_s:10.6f}  {group[-1].toa_s:10.6f}  "
            f"{max(amps):9.2f}  {sum(amps)/len(amps):9.2f}  {','.join(mods)}"
        )

    print("\nFirst 30 measured E3-candidate PDWs:")
    for pdw in e3_pdws[:30]:
        print(pdw)

    # Recover illumination visits from the PDWs themselves.  A new visit starts
    # after a sufficiently long gap in detected 9.410 GHz pulses.  This is only a
    # diagnostic grouping; the behaviour tracker will later consume the PDW stream.
    visits = []
    current = []
    gap_s = 0.20
    for pdw in e3_pdws:
        if current and pdw.toa_s - current[-1].toa_s > gap_s:
            visits.append(current)
            current = []
        current.append(pdw)
    if current:
        visits.append(current)

    print("\nDetected illumination visits derived from measured PDWs")
    print("visit  start(s)   end(s)    duration(ms)  PDWs  peak(dBFS)  dominant waveform")
    print("-----  --------  --------  ------------  ----  ----------  -----------------")
    visit_centres = []
    for i, visit in enumerate(visits, 1):
        start = visit[0].toa_s
        end = visit[-1].toa_s
        centre = 0.5 * (start + end)
        visit_centres.append(centre)
        amps = [p.amplitude_dbfs for p in visit]
        counts = defaultdict(int)
        for p in visit:
            counts[p.modulation_type] += 1
        dominant = max(counts, key=counts.get)
        print(f"{i:5d}  {start:8.4f}  {end:8.4f}  {(end-start)*1e3:12.2f}  {len(visit):4d}  {max(amps):10.2f}  {dominant}")

    intervals = [b-a for a,b in zip(visit_centres, visit_centres[1:])]
    scan_like = [x for x in intervals if 1.0 <= x <= 3.0]
    print("\nMeasured-stream summary")
    print(f"  All measured PDWs       : {all_count}")
    print(f"  9.410 GHz candidates    : {len(e3_pdws)}")
    print(f"  Illumination visits     : {len(visits)}")
    if scan_like:
        period = sum(scan_like) / len(scan_like)
        print(f"  Mean scan-like interval : {period:.3f} s")
        print(f"  Implied scan rate       : {60.0/period:.2f} RPM")
    else:
        print("  Mean scan-like interval : insufficient measured visits")

    if len(e3_pdws) == 0:
        raise SystemExit("FAIL: no 9.410 GHz PDWs emerged from the common sampled-IQ chain")
    print("\nPASS: E3 candidate observations above are measured PDWs from the common 40 MS/s IQ chain.")


if __name__ == "__main__":
    main()
