from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass
class IlluminationAssessment:
    state: str = "UNRESOLVED"
    scan_period_s: float | None = None
    scan_rate_rpm: float | None = None
    confidence: float = 0.0
    modulation_depth_db: float = 0.0
    continuous_illumination_s: float = 0.0
    last_peak_time_s: float | None = None
    peak_count: int = 0
    system_assessment: str = "UNASSESSED"
    baseline_state: str | None = None
    baseline_confidence: float = 0.0
    recent_change_from: str | None = None
    recent_change_to: str | None = None
    recent_change_time_s: float | None = None
    previous_scan_period_s: float | None = None
    previous_scan_rate_rpm: float | None = None


class EmitterIlluminationTracker:
    """Compact per-physical-emitter amplitude/illumination history.

    The tracker consumes low-rate amplitude observations rather than IQ. One small
    tracker can therefore be maintained for every persistent physical emitter.

    Baseline logic is deliberately conservative:
      * startup learning is UNASSESSED, not CHANGED;
      * PERIODIC_SCAN becomes the baseline only when evidence is > 50%;
      * after a baseline exists, a different resolved state must also exceed 50%
        evidence before the system reports CHANGED;
      * CHANGED is transient. If the new state persists for change_hold_s, it
        becomes the new baseline and the emitter returns to MONITOR.

    Current-state reporting is kept separate from historical evidence. When the
    current observable state is PERSISTENT_ILLUMINATION, scan_period_s and
    scan_rate_rpm are intentionally cleared; the most recent periodic estimate is
    retained in previous_scan_period_s / previous_scan_rate_rpm instead.
    """

    def __init__(
        self,
        history_s=30.0,
        illumination_threshold_db=-8.0,
        persistent_s=1.0,
        peak_separation_s=0.25,
        period_tolerance_fraction=0.15,
        baseline_confidence_threshold=0.50,
        change_confidence_threshold=0.50,
        change_hold_s=5.0,
    ):
        self.history_s = float(history_s)
        self.illumination_threshold_db = float(illumination_threshold_db)
        self.persistent_s = float(persistent_s)
        self.peak_separation_s = float(peak_separation_s)
        self.period_tolerance_fraction = float(period_tolerance_fraction)
        self.baseline_confidence_threshold = float(baseline_confidence_threshold)
        self.change_confidence_threshold = float(change_confidence_threshold)
        self.change_hold_s = float(change_hold_s)

        self.samples = deque()
        self.peaks = deque()
        self._in_illumination = False
        self._illumination_start_s = None
        self._peak_time_s = None
        self._peak_amp_db = -math.inf

        self._baseline_state = None
        self._baseline_confidence = 0.0
        self._candidate_change_state = None
        self._candidate_change_start_s = None
        self._recent_change_from = None
        self._recent_change_to = None
        self._recent_change_time_s = None

        self._last_periodic_period_s = None
        self._last_periodic_rate_rpm = None

    def update(self, time_s, amplitude_db):
        time_s = float(time_s)
        amplitude_db = float(amplitude_db)
        self.samples.append((time_s, amplitude_db))
        cutoff = time_s - self.history_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.peaks and self.peaks[0] < cutoff:
            self.peaks.popleft()

        illuminated = amplitude_db >= self.illumination_threshold_db
        if illuminated and not self._in_illumination:
            self._in_illumination = True
            self._illumination_start_s = time_s
            self._peak_time_s = time_s
            self._peak_amp_db = amplitude_db
        elif illuminated and self._in_illumination:
            if amplitude_db > self._peak_amp_db:
                self._peak_amp_db = amplitude_db
                self._peak_time_s = time_s
        elif not illuminated and self._in_illumination:
            self._finish_illumination()

        return self.assess(time_s)

    def _finish_illumination(self):
        if self._peak_time_s is not None:
            if not self.peaks or self._peak_time_s - self.peaks[-1] >= self.peak_separation_s:
                self.peaks.append(self._peak_time_s)
        self._in_illumination = False
        self._illumination_start_s = None
        self._peak_time_s = None
        self._peak_amp_db = -math.inf

    def _observable_assessment(self, now_s):
        continuous_s = 0.0
        if self._in_illumination and self._illumination_start_s is not None:
            continuous_s = max(0.0, float(now_s) - self._illumination_start_s)

        amps = [a for _, a in self.samples]
        modulation_depth = max(amps) - min(amps) if len(amps) >= 2 else 0.0

        peak_list = list(self.peaks)
        intervals = [b - a for a, b in zip(peak_list, peak_list[1:])]
        period = None
        periodic_confidence = 0.0
        if len(intervals) >= 2:
            period = statistics.median(intervals)
            if period > 0.0:
                deviations = [abs(x - period) / period for x in intervals]
                consistency = max(
                    0.0,
                    1.0
                    - statistics.median(deviations)
                    / self.period_tolerance_fraction,
                )
                evidence = min(1.0, len(intervals) / 4.0)
                periodic_confidence = consistency * evidence

        if continuous_s >= self.persistent_s:
            state = "PERSISTENT_ILLUMINATION"
            confidence = max(
                periodic_confidence,
                min(1.0, continuous_s / (2.0 * self.persistent_s)),
            )
            current_period = None
            current_rpm = None
        elif period is not None and periodic_confidence >= 0.45:
            state = "PERIODIC_SCAN"
            confidence = periodic_confidence
            current_period = period
            current_rpm = 60.0 / period if period > 0.0 else None
            self._last_periodic_period_s = current_period
            self._last_periodic_rate_rpm = current_rpm
        elif len(self.samples) < 3:
            state = "UNRESOLVED"
            confidence = periodic_confidence
            current_period = None
            current_rpm = None
        else:
            state = "INTERMITTENT"
            confidence = periodic_confidence
            current_period = None
            current_rpm = None

        last_peak = self.peaks[-1] if self.peaks else self._peak_time_s
        return {
            "state": state,
            "scan_period_s": current_period,
            "scan_rate_rpm": current_rpm,
            "confidence": confidence,
            "modulation_depth_db": modulation_depth,
            "continuous_illumination_s": continuous_s,
            "last_peak_time_s": last_peak,
            "peak_count": len(self.peaks),
            "previous_scan_period_s": self._last_periodic_period_s,
            "previous_scan_rate_rpm": self._last_periodic_rate_rpm,
        }

    def _update_system_assessment(self, now_s, observable):
        state = observable["state"]
        confidence = observable["confidence"]

        if self._baseline_state is None:
            if (
                state == "PERIODIC_SCAN"
                and confidence > self.baseline_confidence_threshold
            ):
                self._baseline_state = state
                self._baseline_confidence = confidence
                self._candidate_change_state = None
                self._candidate_change_start_s = None
                return "MONITOR"
            return "UNASSESSED"

        if state == self._baseline_state:
            self._baseline_confidence = max(self._baseline_confidence, confidence)
            self._candidate_change_state = None
            self._candidate_change_start_s = None
            return "MONITOR"

        if (
            state in ("UNRESOLVED", "INTERMITTENT")
            or confidence <= self.change_confidence_threshold
        ):
            self._candidate_change_state = None
            self._candidate_change_start_s = None
            return "MONITOR"

        if self._candidate_change_state != state:
            self._candidate_change_state = state
            self._candidate_change_start_s = float(now_s)
            self._recent_change_from = self._baseline_state
            self._recent_change_to = state
            self._recent_change_time_s = float(now_s)
            return "CHANGED"

        if (
            self._candidate_change_start_s is not None
            and float(now_s) - self._candidate_change_start_s >= self.change_hold_s
        ):
            self._baseline_state = state
            self._baseline_confidence = confidence
            self._candidate_change_state = None
            self._candidate_change_start_s = None
            return "MONITOR"

        return "CHANGED"

    def assess(self, now_s=None):
        if now_s is None:
            now_s = self.samples[-1][0] if self.samples else 0.0

        observable = self._observable_assessment(now_s)
        system_assessment = self._update_system_assessment(now_s, observable)

        return IlluminationAssessment(
            **observable,
            system_assessment=system_assessment,
            baseline_state=self._baseline_state,
            baseline_confidence=self._baseline_confidence,
            recent_change_from=self._recent_change_from,
            recent_change_to=self._recent_change_to,
            recent_change_time_s=self._recent_change_time_s,
        )


class IlluminationBehaviourManager:
    """Own one lightweight illumination tracker per persistent emitter ID."""

    def __init__(self, **tracker_kwargs):
        self.tracker_kwargs = tracker_kwargs
        self.trackers = {}

    def update(self, emitter_id, time_s, amplitude_db):
        tracker = self.trackers.get(emitter_id)
        if tracker is None:
            tracker = EmitterIlluminationTracker(**self.tracker_kwargs)
            self.trackers[emitter_id] = tracker
        return tracker.update(time_s, amplitude_db)

    def assessment(self, emitter_id, now_s=None):
        tracker = self.trackers.get(emitter_id)
        return IlluminationAssessment() if tracker is None else tracker.assess(now_s)
