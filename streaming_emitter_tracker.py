"""Small persistent emitter tracker for the streaming demonstrator.

This is intentionally a first tracker, not the final MHT. It associates measured
PDWs using receiver face plus RF proximity and maintains emitter records while
detections are absent. No scenario emitter IDs or mode truth enter this module.

A new emitter is now confirmed only after several mutually consistent PDWs. A
single bad frequency estimate or a short composite/noise event therefore remains
tentative instead of immediately becoming E4/E5 on the operator display.
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
    receiver_face: int = 0
    pdws: deque = field(default_factory=lambda: deque(maxlen=4000))
    total_pulses: int = 0

    def update(self, pdw):
        self.pdws.append(pdw)
        self.total_pulses += 1
        self.last_seen_s = pdw.toa_s
        if not self.receiver_face and getattr(pdw, "receiver_face", 0):
            self.receiver_face = int(pdw.receiver_face)
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
            "receiver_face": self.receiver_face,
        }


@dataclass
class TentativeEmitter:
    frequency_hz: float
    first_seen_s: float
    last_seen_s: float
    receiver_face: int = 0
    pdws: list = field(default_factory=list)

    def update(self, pdw):
        self.pdws.append(pdw)
        self.last_seen_s = pdw.toa_s
        self.frequency_hz = 0.8 * self.frequency_hz + 0.2 * pdw.frequency_hz


class StreamingEmitterTracker:
    def __init__(self, frequency_gate_hz=2_000_000.0,
                 confirmation_pdws=5, tentative_timeout_s=0.250):
        self.frequency_gate_hz = float(frequency_gate_hz)
        self.confirmation_pdws = int(confirmation_pdws)
        self.tentative_timeout_s = float(tentative_timeout_s)
        self.tracks = []
        self.tentative = []
        self.next_id = 1

    def reset(self):
        self.tracks.clear()
        self.tentative.clear()
        self.next_id = 1

    @staticmethod
    def _same_face_values(face_a, face_b):
        a = int(face_a or 0)
        b = int(face_b or 0)
        return a == 0 or b == 0 or a == b

    @classmethod
    def _same_face(cls, pdw, track):
        return cls._same_face_values(
            getattr(pdw, "receiver_face", 0),
            getattr(track, "receiver_face", 0),
        )

    def _promote(self, tentative):
        track = StreamingEmitterTrack(
            emitter_id=f"E{self.next_id}",
            frequency_hz=tentative.frequency_hz,
            first_seen_s=tentative.first_seen_s,
            last_seen_s=tentative.last_seen_s,
            receiver_face=tentative.receiver_face,
        )
        self.next_id += 1
        for pdw in tentative.pdws:
            track.update(pdw)
        self.tracks.append(track)
        return track

    def _expire_tentatives(self, now_s):
        self.tentative = [
            t for t in self.tentative
            if now_s - t.last_seen_s <= self.tentative_timeout_s
        ]

    def update(self, pdws):
        for pdw in sorted(pdws, key=lambda p: p.toa_s):
            self._expire_tentatives(pdw.toa_s)

            candidates = [
                t for t in self.tracks
                if self._same_face(pdw, t)
                and abs(pdw.frequency_hz - t.frequency_hz) <= self.frequency_gate_hz
            ]
            if candidates:
                track = min(candidates, key=lambda t: abs(pdw.frequency_hz-t.frequency_hz))
                track.update(pdw)
                continue

            pf = int(getattr(pdw, "receiver_face", 0) or 0)
            tcandidates = [
                t for t in self.tentative
                if self._same_face_values(pf, t.receiver_face)
                and abs(pdw.frequency_hz - t.frequency_hz) <= self.frequency_gate_hz
            ]
            if tcandidates:
                tentative = min(tcandidates, key=lambda t: abs(pdw.frequency_hz-t.frequency_hz))
                tentative.update(pdw)
            else:
                tentative = TentativeEmitter(
                    frequency_hz=pdw.frequency_hz,
                    first_seen_s=pdw.toa_s,
                    last_seen_s=pdw.toa_s,
                    receiver_face=pf,
                    pdws=[pdw],
                )
                self.tentative.append(tentative)

            if len(tentative.pdws) >= self.confirmation_pdws:
                self._promote(tentative)
                self.tentative.remove(tentative)

        return self.tracks
