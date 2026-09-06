"""Small persistent emitter tracker for the streaming demonstrator.

This is intentionally a first tracker, not the final MHT.  It associates only
measured PDWs using RF proximity and maintains emitter records while detections
are absent.  No scenario emitter IDs or mode truth enter this module.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import statistics


@dataclass
class StreamingEmitterTrack:
    emitter_id: str
    frequency_hz: float
    first_seen_s: float
    last_seen_s: float
    pdws: deque = field(default_factory=lambda: deque(maxlen=4000))
    total_pulses: int = 0

    def update(self, pdw):
        self.pdws.append(pdw)
        self.total_pulses += 1
        self.last_seen_s = pdw.toa_s
        # Slow centroid prevents individual noisy edge PDWs moving the track.
        self.frequency_hz = 0.98 * self.frequency_hz + 0.02 * pdw.frequency_hz

    def summary(self):
        items = list(self.pdws)
        recent = items[-min(300, len(items)):]
        frequencies = [p.frequency_hz for p in recent]
        widths = [p.pulse_width_s for p in recent]
        amplitudes = [p.amplitude_dbfs for p in recent]
        mods = Counter(p.modulation_type for p in recent if p.modulation_type != "UNKNOWN")
        modulation = mods.most_common(1)[0][0] if mods else "UNKNOWN"

        pri_s = None
        if len(recent) >= 3:
            toas = [p.toa_s for p in recent]
            diffs = [b-a for a,b in zip(toas,toas[1:]) if b>a]
            # Median works for the stable first demonstration; missed-pulse-aware
            # PRI estimation belongs in the later MHT tracker.
            if diffs:
                pri_s = statistics.median(diffs)

        good_widths = [w for w in widths if w >= 2.0e-6]
        return {
            "frequency_hz": statistics.median(frequencies) if frequencies else self.frequency_hz,
            "pulse_width_s": statistics.median(good_widths or widths) if widths else 0.0,
            "amplitude_dbfs": max(amplitudes) if amplitudes else -120.0,
            "modulation": modulation,
            "pri_s": pri_s,
            "pri_pattern": "STABLE" if pri_s is not None else "UNRESOLVED",
            "pulse_count": self.total_pulses,
        }


class StreamingEmitterTracker:
    def __init__(self, frequency_gate_hz=2_000_000.0):
        self.frequency_gate_hz = float(frequency_gate_hz)
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def update(self, pdws):
        for pdw in sorted(pdws, key=lambda p: p.toa_s):
            candidates = [
                t for t in self.tracks
                if abs(pdw.frequency_hz - t.frequency_hz) <= self.frequency_gate_hz
            ]
            if candidates:
                track = min(candidates, key=lambda t: abs(pdw.frequency_hz-t.frequency_hz))
            else:
                track = StreamingEmitterTrack(
                    emitter_id=f"E{self.next_id}", frequency_hz=pdw.frequency_hz,
                    first_seen_s=pdw.toa_s, last_seen_s=pdw.toa_s,
                )
                self.next_id += 1
                self.tracks.append(track)
            track.update(pdw)
        return self.tracks
