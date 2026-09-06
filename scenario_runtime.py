"""Runtime for scripted S2B emitter behaviour scenarios.

This module deliberately separates simulator truth (emitter type and mode names)
from the observations that will later be presented to the ESM.  The ESM should
receive generated RF/antenna consequences, never the active mode label itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Scripted scenarios require PyYAML. Install it with: pip install pyyaml"
    ) from exc


@dataclass(frozen=True)
class EmitterRuntimeState:
    emitter_id: str
    emitter_type: str
    aoa_deg: float
    peak_level_dbfs: float
    active_mode: str
    mode: dict[str, Any]
    event_time_s: float


class ScenarioRuntime:
    def __init__(self, emitter_types_path, scenario_path):
        self.emitter_types_path = Path(emitter_types_path)
        self.scenario_path = Path(scenario_path)
        self.emitter_types = self._load_yaml(self.emitter_types_path)
        self.document = self._load_yaml(self.scenario_path)
        self.scenario = self.document.get("scenario", {})
        self.emitters = self.document.get("emitters", [])
        self._validate()

    @staticmethod
    def _load_yaml(path):
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return data

    @property
    def name(self):
        return str(self.scenario["name"])

    @property
    def duration_s(self):
        return float(self.scenario["duration_s"])

    def _validate(self):
        if "name" not in self.scenario or "duration_s" not in self.scenario:
            raise ValueError("Scenario requires name and duration_s")
        if float(self.scenario["duration_s"]) <= 0.0:
            raise ValueError("Scenario duration_s must be positive")
        if not isinstance(self.emitters, list) or not self.emitters:
            raise ValueError("Scenario requires at least one emitter")

        ids = set()
        for emitter in self.emitters:
            emitter_id = str(emitter.get("id", ""))
            emitter_type = str(emitter.get("type", ""))
            if not emitter_id:
                raise ValueError("Every emitter requires an id")
            if emitter_id in ids:
                raise ValueError(f"Duplicate emitter id: {emitter_id}")
            ids.add(emitter_id)
            if emitter_type not in self.emitter_types:
                raise ValueError(f"{emitter_id}: unknown emitter type {emitter_type}")

            modes = self.emitter_types[emitter_type].get("modes", {})
            timeline = emitter.get("timeline", [])
            if not timeline:
                raise ValueError(f"{emitter_id}: timeline cannot be empty")
            times = [float(event["time_s"]) for event in timeline]
            if times[0] != 0.0:
                raise ValueError(f"{emitter_id}: first timeline event must start at 0 s")
            if times != sorted(times) or len(times) != len(set(times)):
                raise ValueError(f"{emitter_id}: timeline times must be unique and increasing")
            for event in timeline:
                mode_name = str(event.get("mode", ""))
                if mode_name not in modes:
                    raise ValueError(f"{emitter_id}: unknown mode {mode_name}")

    def emitter_ids(self):
        return [str(emitter["id"]) for emitter in self.emitters]

    def state(self, emitter_id, time_s):
        time_s = float(time_s)
        emitter = next(
            (item for item in self.emitters if str(item["id"]) == str(emitter_id)),
            None,
        )
        if emitter is None:
            raise KeyError(f"Unknown emitter id: {emitter_id}")

        event = emitter["timeline"][0]
        for candidate in emitter["timeline"]:
            if float(candidate["time_s"]) <= time_s:
                event = candidate
            else:
                break

        emitter_type = str(emitter["type"])
        mode_name = str(event["mode"])
        mode = dict(self.emitter_types[emitter_type]["modes"][mode_name])
        return EmitterRuntimeState(
            emitter_id=str(emitter["id"]),
            emitter_type=emitter_type,
            aoa_deg=float(emitter.get("aoa_deg", 0.0)),
            peak_level_dbfs=float(emitter.get("peak_level_dbfs", -6.0)),
            active_mode=mode_name,
            mode=mode,
            event_time_s=float(event["time_s"]),
        )

    def states(self, time_s):
        return [self.state(emitter_id, time_s) for emitter_id in self.emitter_ids()]


def _format_state(state):
    antenna = state.mode.get("antenna", {})
    antenna_text = antenna.get("type", "UNKNOWN")
    if antenna_text == "ROTATING":
        antenna_text += f" {float(antenna.get('rpm', 0.0)):.1f} RPM"
    return (
        f"{state.emitter_id}: mode={state.active_mode:<8} "
        f"AOA={state.aoa_deg:6.1f} deg  "
        f"RF={float(state.mode['frequency_hz']) / 1e9:.3f} GHz  "
        f"PRI={float(state.mode['pri_us']):7.1f} us  "
        f"PW={float(state.mode['pw_us']):5.1f} us  "
        f"WF={state.mode['waveform']:<4}  ANT={antenna_text}"
    )


def main():
    root = Path(__file__).resolve().parent
    runtime = ScenarioRuntime(
        root / "emitter_types.yaml",
        root / "scripted_scenarios" / "nav_scan_to_dwell.yaml",
    )
    print(f"Scenario: {runtime.name}  duration={runtime.duration_s:.1f} s")
    print("Simulator truth below is for scenario validation only; it is not ESM input.\n")
    for time_s in (0.0, 10.0, 14.9, 15.0, 20.0, 24.9, 25.0, 30.0):
        print(f"t={time_s:5.1f} s")
        for state in runtime.states(time_s):
            print("  " + _format_state(state))


if __name__ == "__main__":
    main()
