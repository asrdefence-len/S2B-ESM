"""First long-timescale S2B illumination-behaviour experiment.

This deliberately operates above the raw 40 MS/s IQ simulation so we can model
seconds of antenna motion cheaply. It represents the observations that the PDW /
physical-emitter layer will eventually feed to the same behaviour tracker.
"""

import math
import random

from beam_model import RotatingSincBeam
from illumination_behaviour import EmitterIlluminationTracker


RADAR_FREQUENCY_HZ = 9.410e9
RADAR_BEARING_DEG = 45.0
SCAN_RATE_RPM = 30.0
BEAMWIDTH_DEG = 3.0
PEAK_LEVEL_DBFS = -6.0
OBSERVATION_FLOOR_DBFS = -55.0


def simulated_level_dbfs(beam, time_s, noise_std_db=0.6):
    gain_db = beam.gain_db(RADAR_BEARING_DEG, time_s)
    level = max(OBSERVATION_FLOOR_DBFS, PEAK_LEVEL_DBFS + gain_db)
    return level + random.gauss(0.0, noise_std_db)


def run_demo(duration_s=10.0, update_s=0.01, dwell_after_s=None):
    beam = RotatingSincBeam(
        beamwidth_deg=BEAMWIDTH_DEG,
        scan_rate_rpm=SCAN_RATE_RPM,
        # Arrange first beam crossing near 0.5 s at 45 degrees.
        initial_azimuth_deg=315.0,
        sidelobe_floor_db=-50.0,
    )
    tracker = EmitterIlluminationTracker(
        history_s=30.0,
        illumination_threshold_db=-18.0,
        persistent_s=0.75,
        peak_separation_s=0.5,
    )

    print("S2B ESM NAVIGATION RADAR ILLUMINATION DEMO")
    print("==========================================")
    print(f"RF             : {RADAR_FREQUENCY_HZ/1e9:.3f} GHz")
    print(f"Bearing        : {RADAR_BEARING_DEG:.1f} deg")
    print(f"Beamwidth      : {BEAMWIDTH_DEG:.1f} deg")
    print(f"Truth scan rate: {SCAN_RATE_RPM:.1f} RPM")
    print(f"Truth period   : {60.0/SCAN_RATE_RPM:.3f} s")
    if dwell_after_s is not None:
        print(f"Truth change   : stop scan and dwell on ESM at t={dwell_after_s:.1f} s")
    print()

    last_state = None
    next_report_s = 0.0
    t = 0.0
    while t <= duration_s + 1e-9:
        if dwell_after_s is not None and t >= dwell_after_s:
            beam.fixed_azimuth_deg = RADAR_BEARING_DEG

        level_dbfs = simulated_level_dbfs(beam, t)
        assessment = tracker.update(t, level_dbfs)

        state_changed = assessment.state != last_state
        periodic_report = t >= next_report_s
        if state_changed or periodic_report:
            period_text = "--" if assessment.scan_period_s is None else f"{assessment.scan_period_s:5.3f}s"
            rpm_text = "--" if assessment.scan_rate_rpm is None else f"{assessment.scan_rate_rpm:5.1f}"
            print(
                f"t={t:5.2f}s  level={level_dbfs:6.1f} dBFS  "
                f"{assessment.state:24s}  period={period_text:>7s}  "
                f"rpm={rpm_text:>5s}  conf={100*assessment.confidence:5.1f}%  "
                f"continuous={assessment.continuous_illumination_s:4.2f}s"
            )
            next_report_s = t + 0.5
            last_state = assessment.state

        t += update_s


if __name__ == "__main__":
    # First demonstrate normal 30 RPM scanning. Then run again with dwell_after_s
    # set (for example 6.0) to exercise scan -> persistent illumination.
    run_demo(duration_s=10.0)
