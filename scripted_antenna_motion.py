"""Stateful antenna motion for scripted emitter scenarios.

The antenna position is propagated continuously through mode transitions. A
rotating antenna advances from its current azimuth; a fixed mode holds a defined
pointing angle; when rotation resumes it continues from that held angle rather
than being recomputed from absolute scenario time.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AntennaMotionState:
    antenna_type: str
    azimuth_deg: float
    segment_start_s: float
    mode_name: str


class ScriptedAntennaMotion:
    def __init__(self, runtime, emitter_id):
        self.runtime = runtime
        self.emitter_id = str(emitter_id)
        self.emitter = next(
            item for item in runtime.emitters if str(item["id"]) == self.emitter_id
        )
        self.emitter_type = str(self.emitter["type"])
        self.modes = runtime.emitter_types[self.emitter_type]["modes"]
        self.timeline = list(self.emitter["timeline"])
        self.initial_azimuth_deg = float(
            self.emitter.get("initial_antenna_azimuth_deg", 0.0)
        ) % 360.0
        self.aoa_deg = float(self.emitter.get("aoa_deg", 0.0))

    @staticmethod
    def _wrap360(angle_deg):
        return float(angle_deg) % 360.0

    def _fixed_pointing_deg(self, antenna):
        pointing = antenna.get("pointing", "ESM_BEARING")
        if str(pointing).upper() == "ESM_BEARING":
            return self.aoa_deg % 360.0
        return float(pointing) % 360.0

    def state(self, time_s):
        """Return physical antenna state at scenario time_s.

        Mode transitions are treated as instantaneous for now. In particular,
        entering FIXED slews immediately to the requested pointing direction.
        A later ROTATING mode resumes from that held direction.
        """
        time_s = max(0.0, float(time_s))
        azimuth_deg = self.initial_azimuth_deg

        for index, event in enumerate(self.timeline):
            start_s = float(event["time_s"])
            if start_s > time_s:
                break

            end_s = time_s
            if index + 1 < len(self.timeline):
                end_s = min(time_s, float(self.timeline[index + 1]["time_s"]))

            mode_name = str(event["mode"])
            mode = self.modes[mode_name]
            antenna = mode.get("antenna", {})
            antenna_type = str(antenna.get("type", "UNKNOWN")).upper()

            if antenna_type == "ROTATING":
                rpm = float(antenna.get("rpm", 0.0))
                duration_s = max(0.0, end_s - start_s)
                azimuth_deg = self._wrap360(azimuth_deg + rpm * 6.0 * duration_s)
            elif antenna_type == "FIXED":
                azimuth_deg = self._fixed_pointing_deg(antenna)
            else:
                raise ValueError(
                    f"{self.emitter_id}: unsupported antenna type {antenna_type}"
                )

            if time_s < (float(self.timeline[index + 1]["time_s"]) if index + 1 < len(self.timeline) else float("inf")):
                return AntennaMotionState(
                    antenna_type=antenna_type,
                    azimuth_deg=azimuth_deg,
                    segment_start_s=start_s,
                    mode_name=mode_name,
                )

        # Validation requires a timeline starting at t=0, so this is defensive.
        first_mode = str(self.timeline[0]["mode"])
        first_type = str(self.modes[first_mode].get("antenna", {}).get("type", "UNKNOWN")).upper()
        return AntennaMotionState(
            antenna_type=first_type,
            azimuth_deg=azimuth_deg,
            segment_start_s=0.0,
            mode_name=first_mode,
        )
