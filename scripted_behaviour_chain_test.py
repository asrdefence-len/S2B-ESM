"""End-to-end scripted radar behaviour test at a uniform observation cadence.

Hidden scenario truth drives antenna physics. The behaviour tracker receives only
observable received level relative to the ESM detection threshold; it is never
given the scripted radar mode, scan rate, or transition labels.
"""

import math
from pathlib import Path

from beam_model import RotatingSincBeam
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


def evidence_text(assessment):
    if assessment.state == "PERIODIC_SCAN" and assessment.scan_period_s is not None:
        return f"period={assessment.scan_period_s:.2f}s rate={assessment.scan_rate_rpm:.1f}RPM"
    if assessment.state == "PERSISTENT_ILLUMINATION":
        text = f"duration={assessment.continuous_illumination_s:.2f}s"
        if assessment.previous_scan_period_s is not None:
            text += f" prev_scan={assessment.previous_scan_period_s:.2f}s"
        return text
    return ""


def main():
    root = Path(__file__).resolve().parent
    runtime = ScenarioRuntime(
        root / "emitter_types.yaml",
        root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
    )
    emitter_id = "E3"
    motion = ScriptedAntennaMotion(runtime, emitter_id)
    threshold_dbm = float(runtime.esm_receiver["detection_threshold_dbm"])

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

        important = key != previous_key or detected != previous_detected
        heartbeat = time_s + 1e-9 >= next_summary_s
        if important or heartbeat:
            evidence = evidence_text(assessment)
            print(
                f"t={time_s:5.2f}s  az={antenna_motion.azimuth_deg:6.1f}  "
                f"Prx={prx_dbm:7.1f} dBm  DET={'Y' if detected else 'n'}  "
                f"OBS={assessment.state:<23} SYS={assessment.system_assessment:<10} "
                f"C={assessment.confidence * 100:5.1f}%  {evidence}"
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
    if final.state == "PERIODIC_SCAN" and final.scan_period_s is not None:
        print(f"  Current period   : {final.scan_period_s:.3f} s")
        print(f"  Current rate     : {final.scan_rate_rpm:.2f} RPM")
    elif final.state == "PERSISTENT_ILLUMINATION":
        print(f"  Current duration : {final.continuous_illumination_s:.2f} s")
        if final.previous_scan_period_s is not None:
            print(f"  Previous scan    : {final.previous_scan_period_s:.3f} s")
    if final.recent_change_from and final.recent_change_to:
        print(f"  Recent change    : {final.recent_change_from} -> {final.recent_change_to}")


if __name__ == "__main__":
    main()
