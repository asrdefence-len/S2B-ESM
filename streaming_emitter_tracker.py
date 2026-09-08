"""Small persistent emitter tracker for the streaming demonstrator.

This is intentionally a first tracker, not the final MHT. It associates measured
PDWs using receiver face plus RF proximity and maintains emitter records while
detections are absent. No scenario emitter IDs or mode truth enter this module.

A new emitter is confirmed only after several mutually consistent PDWs. PRI is
estimated as a *track state*, not as the raw interval between arbitrary adjacent
PDWs. Missed pulses therefore appear as integer multiples of the current PRI and
do not by themselves create a new PRI state or emitter identity.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import statistics


MAX_SIGNAL_PRI_S = 0.010
PRI_WINDOW_PDWS = 50
PRI_REL_TOL = 0.08
PRI_ABS_TOL_S = 15e-6
PRI_CHANGE_CONFIRMATIONS = 3
MIN_PRI_CANDIDATE_S = 2e-6
PRI_TO_PW_MIN_RATIO = 3.0


def estimate_track_pri(pdws, max_pri_s=MAX_SIGNAL_PRI_S):
    """Estimate the fundamental observed PRI for one emitter track.

    Candidate PRIs are taken only from intervals that were actually observed.
    Each candidate is then rewarded when other DTOAs are either close to the
    candidate or close to an integer multiple of it. This suppresses 2T/3T/4T
    gaps caused by missed pulses without inventing an unobserved sub-harmonic.

    A candidate must also be physically compatible with the measured pulse
    width. A detector artefact that produces several PDWs inside one 5--7 us
    pulse must never be accepted as a 2--5 us radar PRI.

    Returns (pri_s, confidence) or (None, 0.0).
    """
    items = list(pdws)[-PRI_WINDOW_PDWS:]
    if len(items) < 4:
        return None, 0.0

    toas = [float(p.toa_s) for p in items]
    gaps = [
        b - a for a, b in zip(toas, toas[1:])
        if math.isfinite(a) and math.isfinite(b) and 0.0 < (b - a) <= max_pri_s
    ]
    if len(gaps) < 3:
        return None, 0.0

    widths = [
        float(p.pulse_width_s) for p in items
        if math.isfinite(float(p.pulse_width_s)) and float(p.pulse_width_s) > 0.0
    ]
    representative_pw_s = statistics.median(widths) if widths else 0.0
    physical_min_pri_s = max(
        MIN_PRI_CANDIDATE_S,
        PRI_TO_PW_MIN_RATIO * representative_pw_s,
    )

    def close(a, b):
        return b > 0.0 and abs(a - b) <= max(PRI_ABS_TOL_S, PRI_REL_TOL * b)

    quantum = 2e-6
    candidates = sorted(set(
        q for q in (round(g / quantum) * quantum for g in gaps)
        if math.isfinite(q) and q >= physical_min_pri_s
    ))
    if not candidates:
        return None, 0.0

    best = None

    for candidate in candidates:
        if candidate <= 0.0:
            continue
        direct = 0
        explained = 0
        residual_sum = 0.0
        for gap in gaps:
            # Sub-PW gaps are detector/association artefacts, not useful PRI
            # evidence, and are deliberately excluded from the score.
            if gap < physical_min_pri_s:
                continue
            ratio = max(1, int(round(gap / candidate)))
            predicted = ratio * candidate
            tol = max(PRI_ABS_TOL_S, PRI_REL_TOL * candidate)
            residual = abs(gap - predicted)
            if residual <= tol:
                explained += 1
                residual_sum += residual / tol
                if ratio == 1 and close(gap, candidate):
                    direct += 1

        usable_gaps = sum(g >= physical_min_pri_s for g in gaps)
        if direct < 2 or usable_gaps < 3:
            continue
        explained_fraction = explained / usable_gaps
        direct_fraction = direct / usable_gaps
        mean_residual = residual_sum / explained if explained else 1.0
        score = explained_fraction + 0.35 * direct_fraction - 0.10 * mean_residual
        record = (score, explained_fraction, direct, -candidate, candidate)
        if best is None or record > best:
            best = record

    if best is None:
        return None, 0.0

    _, explained_fraction, direct, _, candidate = best
    confidence = min(1.0, 0.7 * explained_fraction + 0.3 * min(1.0, direct / 6.0))
    return float(candidate), float(confidence)


@dataclass
class StreamingEmitterTrack:
    emitter_id: str
    frequency_hz: float
    first_seen_s: float
    last_seen_s: float
    receiver_face: int = 0
    pdws: deque = field(default_factory=lambda: deque(maxlen=4000))
    total_pulses: int = 0
    current_pri_s: float = None
    pri_confidence: float = 0.0
    candidate_pri_s: float = None
    candidate_pri_count: int = 0
    pri_state_history: deque = field(default_factory=lambda: deque(maxlen=2000))

    @staticmethod
    def _same_pri(a, b):
        if a is None or b is None or a <= 0.0 or b <= 0.0:
            return False
        return abs(a - b) <= max(PRI_ABS_TOL_S, PRI_REL_TOL * b)

    def _update_pri_state(self, pdw):
        estimate, confidence = estimate_track_pri(self.pdws)
        if estimate is None:
            return

        if self.current_pri_s is None:
            self.current_pri_s = estimate
            self.pri_confidence = confidence
            self.pri_state_history.append((pdw.toa_s, estimate, pdw.pulse_width_s, confidence))
            return

        if self._same_pri(estimate, self.current_pri_s):
            self.current_pri_s = 0.85 * self.current_pri_s + 0.15 * estimate
            self.pri_confidence = confidence
            self.candidate_pri_s = None
            self.candidate_pri_count = 0
            self.pri_state_history.append((pdw.toa_s, self.current_pri_s, pdw.pulse_width_s, confidence))
            return

        if self._same_pri(estimate, self.candidate_pri_s):
            self.candidate_pri_count += 1
        else:
            self.candidate_pri_s = estimate
            self.candidate_pri_count = 1

        if self.candidate_pri_count >= PRI_CHANGE_CONFIRMATIONS:
            self.current_pri_s = self.candidate_pri_s
            self.pri_confidence = confidence
            self.candidate_pri_s = None
            self.candidate_pri_count = 0

        self.pri_state_history.append((pdw.toa_s, self.current_pri_s, pdw.pulse_width_s, self.pri_confidence))

    def update(self, pdw):
        self.pdws.append(pdw)
        self.total_pulses += 1
        self.last_seen_s = pdw.toa_s
        if not self.receiver_face and getattr(pdw, "receiver_face", 0):
            self.receiver_face = int(pdw.receiver_face)
        self.frequency_hz = 0.98 * self.frequency_hz + 0.02 * pdw.frequency_hz
        self._update_pri_state(pdw)

    def summary(self):
        items = list(self.pdws)
        recent = items[-min(300, len(items)):]
        frequencies = [p.frequency_hz for p in recent]
        widths = [p.pulse_width_s for p in recent]
        amplitudes = [p.amplitude_dbfs for p in recent]
        mods = Counter(p.modulation_type for p in recent if p.modulation_type != "UNKNOWN")
        modulation = mods.most_common(1)[0][0] if mods else "UNKNOWN"

        good_widths = [w for w in widths if w >= 2.0e-6]
        return {
            "frequency_hz": statistics.median(frequencies) if frequencies else self.frequency_hz,
            "pulse_width_s": statistics.median(good_widths or widths) if widths else 0.0,
            "amplitude_dbfs": max(amplitudes) if amplitudes else -120.0,
            "modulation": modulation,
            "pri_s": self.current_pri_s,
            "pri_confidence": self.pri_confidence,
            "pri_pattern": "STABLE" if self.current_pri_s is not None else "UNRESOLVED",
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
