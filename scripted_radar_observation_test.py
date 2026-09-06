"""Validate physical antenna motion and the resulting ESM received level.

The scenario defines radar modes and antenna behaviour. This diagnostic derives
antenna azimuth, angular offset, beam gain and received level from that physical
state. No ESM illumination times are scripted.
"""

from pathlib import Path

from beam_model import RotatingSincBeam, wrap_angle_deg
from scenario_runtime import ScenarioRuntime


OBSERVATION_FLOOR_DBFS = -55.0


def antenna_state(runtime_state, time_s):
    antenna = runtime_state.mode.get("antenna", {})
    antenna_type = str(antenna.get("type", "UNKNOWN")).upper()
    beamwidth_deg = float(antenna.get("beamwidth_deg", 3.0))
    pattern = str(antenna.get("pattern", "SINC")).upper()

    if pattern != "SINC":
        raise ValueError(f"Unsupported antenna pattern for this test: {pattern}")

    if antenna_type == "ROTATING":
        rpm = float(antenna.get("rpm", 0.0))
        beam = RotatingSincBeam(
            beamwidth_deg=beamwidth_deg,
            scan_rate_rpm=rpm,
            initial_azimuth_deg=float(runtime_state.initial_antenna_azimuth_deg),
            sidelobe_floor_db=-50.0,
        )
        azimuth_deg = beam.azimuth_deg(time_s)
        gain_db = beam.gain_db(runtime_state.aoa_deg, time_s)
        return antenna_type, azimuth_deg, gain_db

    if antenna_type == "FIXED":
        pointing = antenna.get("pointing", "ESM_BEARING")
        if str(pointing).upper() == "ESM_BEARING":
            pointing_deg = runtime_state.aoa_deg
        else:
            pointing_deg = float(pointing)
        beam = RotatingSincBeam(
            beamwidth_deg=beamwidth_deg,
            scan_rate_rpm=0.0,
            fixed_azimuth_deg=pointing_deg,
            sidelobe_floor_db=-50.0,
        )
        azimuth_deg = beam.azimuth_deg(time_s)
        gain_db = beam.gain_db(runtime_state.aoa_deg, time_s)
        return antenna_type, azimuth_deg, gain_db

    raise ValueError(f"Unsupported antenna type: {antenna_type}")


def received_level_dbfs(runtime_state, gain_db):
    return max(OBSERVATION_FLOOR_DBFS, runtime_state.peak_level_dbfs + gain_db)


def main():
    root = Path(__file__).resolve().parent
    runtime = ScenarioRuntime(
        root / "emitter_types.yaml",
        root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
    )

    emitter_id = "E3"
    print("S2B SCRIPTED RADAR PHYSICAL OBSERVATION TEST")
    print("===========================================")
    print("Scenario truth drives radar/antenna state. Illumination is derived from beam geometry.")
    print()

    # Dense samples around one early beam crossing, plus samples around mode changes.
    sample_times = [
        0.00, 0.40, 0.60, 0.63, 0.64, 0.65, 0.70, 1.00,
        2.60, 2.63, 2.64, 2.65,
        14.90, 14.99, 15.00, 15.10, 16.00,
        24.90, 24.99, 25.00, 25.10, 26.00,
    ]

    print("time    mode      ant       ant_az   delta     gain      level")
    print("s                             deg      deg       dB        dBFS")
    print("----------------------------------------------------------------")

    for time_s in sample_times:
        state = runtime.state(emitter_id, time_s)
        antenna_type, azimuth_deg, gain_db = antenna_state(state, time_s)
        delta_deg = wrap_angle_deg(state.aoa_deg - azimuth_deg)
        level_dbfs = received_level_dbfs(state, gain_db)
        print(
            f"{time_s:5.2f}  {state.active_mode:<8}  {antenna_type:<8}  "
            f"{azimuth_deg:7.2f}  {delta_deg:7.2f}  {gain_db:8.2f}  {level_dbfs:8.2f}"
        )

    print()
    print("Expected behaviour:")
    print("  * During NAV_SCAN, antenna azimuth rotates continuously at 30 RPM.")
    print("  * Received level rises and falls only when the 3 deg sinc beam crosses AOA=135 deg.")
    print("  * At 15 s, DWELL fixes the antenna on the ESM bearing, so received level stays high.")
    print("  * At 25 s, NAV_SCAN resumes and periodic beam crossings return.")


if __name__ == "__main__":
    main()
