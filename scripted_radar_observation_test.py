"""Validate physical antenna motion and resulting ESM received power.

Scenario truth defines radar, geometry and antenna behaviour. Received power and
detection emerge from the one-way RF link budget; illumination times are never
scripted into the ESM.
"""

import math
from pathlib import Path

from beam_model import RotatingSincBeam, wrap_angle_deg
from scenario_runtime import ScenarioRuntime
from scripted_antenna_motion import ScriptedAntennaMotion


C_MPS = 299_792_458.0


def watts_to_dbm(power_w):
    return 10.0 * math.log10(float(power_w) * 1000.0)


def free_space_path_loss_db(frequency_hz, range_km):
    wavelength_m = C_MPS / float(frequency_hz)
    range_m = float(range_km) * 1000.0
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)


def antenna_state(runtime_state, motion_state, time_s):
    antenna = runtime_state.mode.get("antenna", {})
    antenna_type = motion_state.antenna_type
    beamwidth_deg = float(antenna.get("beamwidth_deg", 3.0))
    pattern = str(antenna.get("pattern", "SINC")).upper()

    if pattern != "SINC":
        raise ValueError(f"Unsupported antenna pattern for this test: {pattern}")

    # Motion is handled by ScriptedAntennaMotion. Here the beam is fixed at the
    # already-propagated physical azimuth solely to evaluate the pattern gain.
    beam = RotatingSincBeam(
        beamwidth_deg=beamwidth_deg,
        scan_rate_rpm=0.0,
        fixed_azimuth_deg=motion_state.azimuth_deg,
        sidelobe_floor_db=-50.0,
    )
    relative_pattern_db = beam.gain_db(runtime_state.aoa_deg, time_s)
    return antenna_type, motion_state.azimuth_deg, relative_pattern_db


def received_power_dbm(runtime, runtime_state, relative_pattern_db):
    antenna = runtime_state.mode["antenna"]
    frequency_hz = float(runtime_state.mode["frequency_hz"])
    tx_dbm = watts_to_dbm(runtime_state.tx_peak_power_w)
    tx_peak_gain_dbi = float(antenna.get("peak_gain_dbi", 0.0))
    esm_gain_dbi = float(runtime.esm_receiver.get("antenna_gain_dbi", 0.0))
    fspl_db = free_space_path_loss_db(frequency_hz, runtime_state.range_km)
    return tx_dbm + tx_peak_gain_dbi + relative_pattern_db - fspl_db + esm_gain_dbi


def main():
    root = Path(__file__).resolve().parent
    runtime = ScenarioRuntime(
        root / "emitter_types.yaml",
        root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
    )

    emitter_id = "E3"
    motion = ScriptedAntennaMotion(runtime, emitter_id)
    noise_dbm = float(runtime.esm_receiver["noise_floor_dbm"])
    threshold_dbm = float(runtime.esm_receiver["detection_threshold_dbm"])
    state0 = runtime.state(emitter_id, 0.0)

    print("S2B SCRIPTED RADAR PHYSICAL OBSERVATION TEST")
    print("===========================================")
    print("Illumination is derived from continuous antenna motion, propagation and ESM threshold.")
    print(
        f"{emitter_id}: range={state0.range_km:.1f} km, "
        f"Tx={state0.tx_peak_power_w:.0f} W ({watts_to_dbm(state0.tx_peak_power_w):.1f} dBm), "
        f"ESM gain={float(runtime.esm_receiver['antenna_gain_dbi']):.1f} dBi"
    )
    print(f"ESM noise floor={noise_dbm:.1f} dBm, detection threshold={threshold_dbm:.1f} dBm")
    print()

    sample_times = [
        0.00, 0.60, 0.62, 0.63, 0.635, 0.639, 0.643, 0.65, 0.66, 0.70, 1.00,
        2.62, 2.63, 2.635, 2.639, 2.643, 2.65, 2.66,
        14.90, 14.99, 15.00, 15.10, 16.00,
        24.90, 24.99, 25.00, 25.01, 25.02, 25.05, 25.10, 25.50, 26.00,
    ]

    print("time    mode      ant       ant_az   delta   pattern      Prx    SNR   DET")
    print("s                             deg      deg       dB        dBm     dB")
    print("----------------------------------------------------------------------------")

    for time_s in sample_times:
        state = runtime.state(emitter_id, time_s)
        motion_state = motion.state(time_s)
        antenna_type, azimuth_deg, pattern_db = antenna_state(state, motion_state, time_s)
        delta_deg = wrap_angle_deg(state.aoa_deg - azimuth_deg)
        prx_dbm = received_power_dbm(runtime, state, pattern_db)
        snr_db = prx_dbm - noise_dbm
        detected = prx_dbm >= threshold_dbm
        print(
            f"{time_s:6.3f}  {state.active_mode:<8}  {antenna_type:<8}  "
            f"{azimuth_deg:7.2f}  {delta_deg:7.2f}  {pattern_db:8.2f}  "
            f"{prx_dbm:8.2f}  {snr_db:6.1f}   {'YES' if detected else 'no'}"
        )

    frequency_hz = float(state0.mode["frequency_hz"])
    fspl_db = free_space_path_loss_db(frequency_hz, state0.range_km)
    peak_gain_dbi = float(state0.mode["antenna"]["peak_gain_dbi"])
    peak_prx = received_power_dbm(runtime, state0, 0.0)
    print()
    print("Link budget at main-beam boresight:")
    print(f"  Tx peak power       {watts_to_dbm(state0.tx_peak_power_w):8.2f} dBm")
    print(f"  Tx peak gain        {peak_gain_dbi:8.2f} dBi")
    print(f"  Free-space loss    {-fspl_db:8.2f} dB")
    print(f"  ESM antenna gain    {float(runtime.esm_receiver['antenna_gain_dbi']):8.2f} dBi")
    print(f"  Peak received       {peak_prx:8.2f} dBm")
    print(f"  Threshold margin    {peak_prx - threshold_dbm:8.2f} dB")
    print()
    print("Expected behaviour:")
    print("  * NAV_SCAN rotates continuously from the antenna's current physical azimuth.")
    print("  * At 15 s DWELL slews to and holds the ESM bearing at 135 deg.")
    print("  * At 25 s NAV_SCAN resumes rotation from 135 deg, not from an absolute-time phase.")


if __name__ == "__main__":
    main()
