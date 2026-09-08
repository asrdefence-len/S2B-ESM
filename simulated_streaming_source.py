"""Single-sector 40 MS/s IQ simulator for the unified S2B ESM front end.

All three demonstration emitters are deliberately placed in the same 90-degree
receiver sector and rendered into ONE complex IQ stream. This keeps the live
processing load at 40 MS/s rather than accidentally simulating four independent
40 MS/s receivers.

Test RF placement:
    E1 legacy Radar A : 9.4200 GHz
    E2 legacy Radar B : 9.4225 GHz
    E3 NAVRADAR       : 9.4100 GHz (from scripted scenario)

E1 and E2 are rotating search radars rather than continuously visible pulse
trains. Their antenna patterns cross the ESM bearing every 5 s and 7 s
respectively. They are placed at the same 15 km range as E3 and use the same
simple propagation/detection scaling, so sidelobes should fall below the detector
threshold and PDWs should appear mainly during beam crossings.

The E1/E2 spacing is intentionally greater than the current 2 MHz lightweight
tracker RF gate. For this simple harness, the E2 start time is shifted by 50 us
relative to the legacy close-emitter scenario so its 1.3 ms train does not land
exactly on the 1.0 ms E1/E3 pulse trains. Exact pulse overlap creates composite
IQ snippets whose single-pulse frequency estimate can legitimately fall between
emitters and seed false tentative tracks; that is a later deinterleaving problem,
not what this basic S2B behaviour demonstration is intended to test.

The scripted scenario duration describes the authored behaviour timeline; it is
not treated as an RF transmitter shutdown. After the final scripted mode change,
E3 continues indefinitely in that final mode so the live ESM can run continuously.
"""

import math
from pathlib import Path

import numpy as np

from beam_model import RotatingSincBeam
from rf_link_budget import received_power_dbm
from scenario_runtime import ScenarioRuntime
from scripted_antenna_motion import ScriptedAntennaMotion
from scenarios import get_scenario

RECEIVER_FACE = 1
RECEIVER_SECTOR_AOA_DEG = 45.0
LEGACY_RF_HZ = (9_420_000_000.0, 9_422_500_000.0)
E2_START_OFFSET_S = 50e-6
LEGACY_SCAN_PERIOD_S = (5.0, 7.0)
LEGACY_SCAN_BEAMWIDTH_DEG = 3.0
LEGACY_SCAN_SIDELOBE_FLOOR_DB = -50.0
LEGACY_RANGE_KM = 15.0
LEGACY_TX_PEAK_POWER_W = 100.0
LEGACY_PEAK_GAIN_DBI = 25.0


class SimulatedStreamingIQSource:
    def __init__(
        self,
        sample_rate_hz=40_000_000,
        center_frequency_hz=9_415_000_000,
        block_samples=40_000,
        noise_std=0.02,
        legacy_scenario_name="close_emitters",
    ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.center_frequency_hz = float(center_frequency_hz)
        self.block_samples = int(block_samples)
        self.noise_std = float(noise_std)
        self.block_duration_s = self.block_samples / self.sample_rate_hz
        self.sample_index = 0
        self.rng = np.random.default_rng(12345)
        self.legacy_scenario = get_scenario(legacy_scenario_name)

        root = Path(__file__).resolve().parent
        self.scripted_runtime = ScenarioRuntime(
            root / "emitter_types.yaml",
            root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
        )
        self.e3_motion = ScriptedAntennaMotion(self.scripted_runtime, "E3")

    @property
    def time_s(self):
        return self.sample_index / self.sample_rate_hz

    def reset(self):
        self.sample_index = 0
        self.rng = np.random.default_rng(12345)

    def _add_pulse(
        self,
        iq,
        block_start_s,
        pulse_toa_s,
        pulse_width_s,
        rf_frequency_hz,
        amplitude,
        modulation="CW",
        bandwidth_hz=0.0,
    ):
        start = int(round((pulse_toa_s - block_start_s) * self.sample_rate_hz))
        pulse_samples = max(1, int(round(pulse_width_s * self.sample_rate_hz)))
        stop = start + pulse_samples
        dst_start = max(0, start)
        dst_stop = min(len(iq), stop)
        if dst_start >= dst_stop:
            return

        src_start = dst_start - start
        count = dst_stop - dst_start
        pulse_t = np.arange(src_start, src_start + count, dtype=float) / self.sample_rate_hz
        if_hz = float(rf_frequency_hz) - self.center_frequency_hz
        if str(modulation).upper() in ("FM", "LFM") and bandwidth_hz > 0.0:
            k = float(bandwidth_hz) / float(pulse_width_s)
            f0 = if_hz - 0.5 * float(bandwidth_hz)
            phase = 2.0 * np.pi * (f0 * pulse_t + 0.5 * k * pulse_t**2)
        else:
            phase = 2.0 * np.pi * if_hz * pulse_t
        iq[dst_start:dst_stop] += (
            float(amplitude) * np.exp(1j * phase)
        ).astype(np.complex64)

    @staticmethod
    def _periodic_toas(start_s, pri_s, block_start_s, block_end_s):
        if block_end_s <= start_s:
            return []
        n0 = max(0, int(math.ceil((block_start_s - start_s) / pri_s - 1e-12)))
        t = start_s + n0 * pri_s
        out = []
        while t < block_end_s - 1e-15:
            if t >= block_start_s - 1e-15:
                out.append(t)
            n0 += 1
            t = start_s + n0 * pri_s
        return out

    @staticmethod
    def _legacy_scan_gain(time_s, scan_period_s):
        """Return rotating antenna gain toward the fixed ESM bearing.

        Important: initial_azimuth_deg sets the starting angle while leaving the
        antenna free to rotate. fixed_azimuth_deg would intentionally disable
        rotation, which is not what E1/E2 require.
        """
        scan_rate_rpm = 60.0 / float(scan_period_s)
        beam = RotatingSincBeam(
            beamwidth_deg=LEGACY_SCAN_BEAMWIDTH_DEG,
            scan_rate_rpm=scan_rate_rpm,
            initial_azimuth_deg=0.0,
            sidelobe_floor_db=LEGACY_SCAN_SIDELOBE_FLOOR_DB,
        )
        return beam.gain_db(0.0, float(time_s))

    def _render_e1_e2(self, iq, block_start_s, block_end_s):
        rx_gain_dbi = float(self.scripted_runtime.esm_receiver["antenna_gain_dbi"])
        threshold_dbm = float(
            self.scripted_runtime.esm_receiver["detection_threshold_dbm"]
        )
        for idx, emitter in enumerate(self.legacy_scenario.emitters):
            rf_hz = LEGACY_RF_HZ[idx % len(LEGACY_RF_HZ)]
            scan_period_s = LEGACY_SCAN_PERIOD_S[idx % len(LEGACY_SCAN_PERIOD_S)]
            start_delay_s = float(emitter["start_delay_s"])
            if idx == 1:
                start_delay_s += E2_START_OFFSET_S
            for toa_s in self._periodic_toas(
                start_delay_s,
                float(emitter["pri_s"]),
                block_start_s,
                block_end_s,
            ):
                pattern_db = self._legacy_scan_gain(toa_s, scan_period_s)
                prx_dbm = received_power_dbm(
                    LEGACY_TX_PEAK_POWER_W,
                    LEGACY_PEAK_GAIN_DBI,
                    pattern_db,
                    rf_hz,
                    LEGACY_RANGE_KM,
                    rx_gain_dbi,
                )
                amplitude = 0.10 * 10.0 ** ((prx_dbm - threshold_dbm) / 20.0)
                amplitude = min(0.90, max(0.0, amplitude))
                self._add_pulse(
                    iq,
                    block_start_s,
                    toa_s,
                    float(emitter["pulse_width_s"]),
                    rf_hz,
                    amplitude,
                    modulation=emitter.get("modulation", "CW"),
                    bandwidth_hz=float(emitter.get("lfm_bandwidth_hz", 0.0)),
                )

    def _e3_mode_segments(self):
        emitter = next(
            e for e in self.scripted_runtime.emitters if str(e["id"]) == "E3"
        )
        timeline = emitter["timeline"]
        segments = []
        for i, event in enumerate(timeline):
            end = (
                float(timeline[i + 1]["time_s"])
                if i + 1 < len(timeline)
                else math.inf
            )
            segments.append((float(event["time_s"]), end, str(event["mode"])))
        return segments

    def _render_e3(self, iq, block_start_s, block_end_s):
        rx_gain_dbi = float(self.scripted_runtime.esm_receiver["antenna_gain_dbi"])
        threshold_dbm = float(
            self.scripted_runtime.esm_receiver["detection_threshold_dbm"]
        )
        for segment_start, segment_end, _ in self._e3_mode_segments():
            start = max(block_start_s, segment_start)
            end = min(block_end_s, segment_end)
            if start >= end:
                continue
            state0 = self.scripted_runtime.state("E3", start)
            pri_s = float(state0.mode["pri_us"]) * 1e-6
            for toa_s in self._periodic_toas(segment_start, pri_s, start, end):
                state = self.scripted_runtime.state("E3", toa_s)
                motion = self.e3_motion.state(toa_s)
                antenna = state.mode["antenna"]
                beam = RotatingSincBeam(
                    beamwidth_deg=float(antenna.get("beamwidth_deg", 3.0)),
                    scan_rate_rpm=0.0,
                    fixed_azimuth_deg=float(motion.azimuth_deg),
                    sidelobe_floor_db=-50.0,
                )
                pattern_db = beam.gain_db(state.aoa_deg, 0.0)
                prx_dbm = received_power_dbm(
                    state.tx_peak_power_w,
                    float(antenna.get("peak_gain_dbi", 0.0)),
                    pattern_db,
                    float(state.mode["frequency_hz"]),
                    state.range_km,
                    rx_gain_dbi,
                )
                amplitude = 0.10 * 10.0 ** ((prx_dbm - threshold_dbm) / 20.0)
                amplitude = min(0.90, max(0.0, amplitude))
                self._add_pulse(
                    iq,
                    block_start_s,
                    toa_s,
                    float(state.mode["pw_us"]) * 1e-6,
                    float(state.mode["frequency_hz"]),
                    amplitude,
                    modulation=state.mode.get("waveform", "CW"),
                    bandwidth_hz=float(state.mode.get("bandwidth_hz", 0.0)),
                )

    def read_block(self):
        block_start_s = self.time_s
        block_end_s = block_start_s + self.block_duration_s
        iq = np.zeros(self.block_samples, dtype=np.complex64)

        self._render_e1_e2(iq, block_start_s, block_end_s)
        self._render_e3(iq, block_start_s, block_end_s)

        if self.noise_std > 0.0:
            noise = (
                self.rng.normal(0.0, self.noise_std, self.block_samples)
                + 1j * self.rng.normal(0.0, self.noise_std, self.block_samples)
            ).astype(np.complex64)
            iq += noise

        metadata = {
            "sample_rate_hz": self.sample_rate_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "start_time_s": block_start_s,
            "sample_index": self.sample_index,
            "receiver_face": RECEIVER_FACE,
            "receiver_sector_aoa_deg": RECEIVER_SECTOR_AOA_DEG,
        }
        self.sample_index += self.block_samples
        return iq, metadata
