from collections import deque


class ObservedModeHistory:
    """One-second cadence rolling history of operator-level mode labels.

    PERIODIC_SCAN maps to SEARCH and PERSISTENT_ILLUMINATION maps to DWELL.
    Unresolved/intermittent illumination is an observation state rather than a
    radar mode, so the ticker holds the last inferred radar mode through it.

    A newly resolved mode may replace the current one-second cell. This matters
    at transitions: if SEARCH was sampled early in a second and DWELL becomes
    established later in that same second, DWELL must not be discarded simply
    because that second already has a ticker entry.
    """

    def __init__(self, max_entries=18):
        self.max_entries = int(max_entries)
        self._history = {}
        self._last_second = {}
        self._last_mode = {}

    @staticmethod
    def label_from_illumination(illumination_state):
        state = str(illumination_state or "").upper()
        if state == "PERIODIC_SCAN":
            return "SEARCH"
        if state == "PERSISTENT_ILLUMINATION":
            return "DWELL"
        return None

    def update(self, emitter_id, time_s, illumination_state):
        eid = str(emitter_id)
        second = int(float(time_s))
        resolved_label = self.label_from_illumination(illumination_state)
        label = resolved_label or self._last_mode.get(eid)
        if label is None:
            return

        q = self._history.setdefault(eid, deque(maxlen=self.max_entries))

        if self._last_second.get(eid) == second:
            # Do not let the first sample in a one-second cell mask a genuine
            # mode transition detected later in that same cell.
            if resolved_label is not None and q and q[-1] != resolved_label:
                q[-1] = resolved_label
                self._last_mode[eid] = resolved_label
            return

        q.append(label)
        self._last_second[eid] = second
        if resolved_label is not None:
            self._last_mode[eid] = resolved_label

    def labels(self, emitter_id):
        return list(self._history.get(str(emitter_id), ()))

    def clear(self):
        self._history.clear()
        self._last_second.clear()
        self._last_mode.clear()
