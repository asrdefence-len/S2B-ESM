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

    Scan inference is deliberately hypothesis-forming rather than excessively
    conservative:
      * two separated beam crossings are enough to form a provisional scan period;
      * a third consistent crossing raises confidence quickly;
      * once a scan period has been learned, later crossings can rapidly reacquire
        the same scan after a dwell instead of learning it from scratch;
      * long persistent dwells are not inserted into the scan-peak history.
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
            self._finish_illumination(time_s)

        return self.assess(time_s)

    def _finish_illumination(self, end_time_s):
        duration_s = 0.0
        if self._illumination_start_s is not None:
            duration_s = max(0.0, float(end_time_s) - self._illumination_start_s)

        # A short illumination burst is a candidate rotating-beam crossing. A
        # long dwell is behavioural evidence in its own right and must not be
        # allowed to corrupt the scan-period peak sequence when the dwell ends.
        if self._peak_time_s is not None and duration_s < self.persistent_s:
            if not self.peaks or self._peak_time_s - self.peaks[-1] >= self.peak_separation_s:
                self.peaks.append(self._peak_time_s)

        self._in_illumination = False
        self._illumination_start_s = None
        self._peak_time_s = None
        self._peak_amp_db = -math.inf

    def _scan_estimate(self):
        peak_list = list(self.peaks)
        intervals = [b - a for a, b in zip(peak_list, peak_list[1:]) if b > a]
        if not intervals:
            return None, 0.0

        # Two crossings give one interval: enough for a useful provisional scan
        # hypothesis. Three or more crossings let consistency drive confidence.
        if len(intervals) == 1:
            interval = float(intervals[-1])
            if self._last_periodic_period_s is not None:
                remembered = float(self._last_periodic_period_s)
                multiple = max(1, int(round(interval / remembered)))
                expected = multiple * remembered
                rel_error = abs(interval - expected) / max(expected, 1e-9)
                if rel_error <= self.period_tolerance_fraction:
                    return remembered, 0.70
            return interval, 0.55

        period = statistics.median(intervals)
        if period <= 0.0:
            return None, 0.0
        deviations = [abs(x - period) / period for x in intervals]
        consistency = max(
            0.0,
            1.0 - statistics.median(deviations) / self.period_tolerance_fraction,
        )
        # Two intervals (three crossings) should already be strong evidence.
        evidence = min(1.0, 0.65 + 0.15 * (len(intervals) - 2))
        confidence = consistency * evidence
        return float(period), float(confidence)

    def _remembered_scan_reacquisition(self):
        """Use a learned scan period to reacquire SEARCH after a dwell quickly."""
        if self._last_periodic_period_s is None or len(self.peaks) < 2:
            return None, 0.0
        p0, p1 = self.peaks[-2], self.peaks[-1]
        gap = p1 - p0
        remembered = float(self._last_periodic_period_s)
        multiple = max(1, int(round(gap / remembered)))
        expected = multiple * remembered
        rel_error = abs(gap - expected) / max(expected, 1e-9)
        if rel_error <= self.period_tolerance_fraction:
            confidence = max(0.65, 1.0 - rel_error / self.period_tolerance_fraction)
            return remembered, min(0.90, confidence)
        return None, 0.0

    def _observable_assessment(self, now_s):
        continuous_s = 0.0
        if self._in_illumination and self._illumination_start_s is not None:
            continuous_s = max(0.0, float(now_s) - self._illumination_start_s)

        amps = [a for _, a in self.samples]
        modulation_depth = max(amps) - min(amps) if len(amps) >= 2 else 0.0

        period, periodic_confidence = self._scan_estimate()
        remembered_period, remembered_confidence = self._remembered_scan_reacquisition()
        if remembered_confidence > periodic_confidence:
            period = remembered_period
            periodic_confidence = remembered_confidence

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
