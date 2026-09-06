"""End-to-end scripted radar behaviour test at a uniform observation cadence.

Hidden scenario truth drives antenna physics. The behaviour tracker receives only
observable received level relative to the ESM detection threshold; it is never
given the scripted radar mode, scan rate, or transition labels.
"""

import math
from pathlib import Path

from beam_model import RotatingSincBeam, wrap_angle_deg
from illumination_behaviour import EmitterIlluminationTracker
from scenario_runtime import ScenarioRuntime
from scripted_antenna_motion import ScriptedAntennaMotion

C_MPS = 299_792_458.0
DT_S = 0.010


def watts_to_dbm(power_w):
    return 10.0 * math.log10(float(power_w) * 1000.0)


def free_space_path_loss_db(frequency_hz, range_km):
    wavelength_m = C_MPS / float(frequency_hz)
    range_m = float(range_km) * 1000.0
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)


def relative_pattern_db(state, antenna_azimuth_deg):
    antenna = state.mode["antenna"]
    beam = RotatingSincBeam(
        beamwidth_deg=float(antenna.get("beamwidth_deg", 3.0)),
        scan_rate_rpm=0.0,
        fixed_azimuth_deg=float(antenna_azimuth_deg),
        sidelobe_floor_db=-50.0,
    )
    return beam.gain_db(state.aoa_deg, 0.0)


def received_power_dbm(runtime, state, pattern_db):
    antenna = state.mode["antenna"]
    tx_dbm = watts_to_dbm(state.tx_peak_power_w)
    tx_gain_dbi = float(antenna.get("peak_gain_dbi", 0.0))
    rx_gain_dbi = float(runtime.esm_receiver.get("antenna_gain_dbi", 0.0))
    fspl_db = free_space_path_loss_db(float(state.mode["frequency_hz"]), state.range_km)
    return tx_dbm + tx_gain_dbi + pattern_db - fspl_db + rx_gain_dbi


def main():
    root = Path(__file__).resolve().parent
    runtime = ScenarioRuntime(
        root / "emitter_types.yaml",
        root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
    )
    emitter_id = "E3"
    motion = ScriptedAntennaMotion(runtime, emitter_id)
    threshold_dbm = float(runtime.esm_receiver["detection_threshold_dbm"])

    # Tracker works in dB relative to the ESM detection threshold. Thus 0 dB is
    # the actual receiver threshold and an illumination is simply a detection.
    tracker = EmitterIlluminationTracker(
        illumination_threshold_db=0.0,
        persistent_s=1.0,
        peak_separation_s=0.25,
        baseline_confidence_threshold=0.50,
        change_confidence_threshold=0.50,
        change_hold_s=5.0,
    )

    print("S2B PHYSICAL RADAR -> ESM -> BEHAVIOUR CHAIN")
    print("============================================")
    print(f"Uniform observation interval: {DT_S * 1000.0:.0f} ms")
    print(f"ESM detection threshold: {threshold_dbm:.1f} dBm")
    print("Behaviour tracker receives Prx-threshold only; scripted mode is not an input.\n")

    previous_key = None
    previous_detected = None
    next_summary_s = 0.0
    time_s = 0.0

    while time_s <= runtime.duration_s + 1e-9:
        state = runtime.state(emitter_id, time_s)
        antenna_motion = motion.state(time_s)
        pattern_db = relative_pattern_db(state, antenna_motion.azimuth_deg)
        prx_dbm = received_power_dbm(runtime, state, pattern_db)
        detected = prx_dbm >= threshold_dbm
        observation_db = prx_dbm - threshold_dbm
        assessment = tracker.update(time_s, observation_db)

        key = (
            assessment.state,
            assessment.system_assessment,
            assessment.baseline_state,
            assessment.recent_change_from,
            assessment.recent_change_to,
        )

        # Print state/assessment transitions, threshold crossings, and a compact
        # 1-second heartbeat. The underlying processing still occurs every 10 ms.
        important = key != previous_key or detected != previous_detected
        heartbeat = time_s + 1e-9 >= next_summary_s
        if important or heartbeat:
            period_text = "-" if assessment.scan_period_s is None else f"{assessment.scan_period_s:4.2f}"
            print(
                f"t={time_s:5.2f}s  az={antenna_motion.azimuth_deg:6.1f}  "
                f"Prx={prx_dbm:7.1f} dBm  DET={'Y' if detected else 'n'}  "
                f"OBS={assessment.state:<23} SYS={assessment.system_assessment:<10} "
                f"P={period_text}s C={assessment.confidence * 100:5.1f}%"
            )
            if heartbeat:
                next_summary_s += 1.0

        previous_key = key
        previous_detected = detected
        time_s = round(time_s + DT_S, 10)

    final = tracker.assess(runtime.duration_s)
    print("\nFinal ESM-derived assessment:")
    print(f"  Observable state : {final.state}")
    print(f"  System assessment: {final.system_assessment}")
    print(f"  Baseline         : {final.baseline_state}")
    if final.scan_period_s is not None:
        print(f"  Estimated period : {final.scan_period_s:.3f} s")
        print(f"  Estimated rate   : {final.scan_rate_rpm:.2f} RPM")
    if final.recent_change_from and final.recent_change_to:
        print(f"  Recent change    : {final.recent_change_from} -> {final.recent_change_to}")


if __name__ == "__main__":
    main()
