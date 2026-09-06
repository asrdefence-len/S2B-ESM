from collections import deque


class ObservedModeHistory:
    """One-second cadence rolling history of observable mode labels per emitter."""

    def __init__(self, max_entries=18):
        self.max_entries = int(max_entries)
        self._history = {}
        self._last_second = {}

    @staticmethod
    def label_from_illumination(illumination_state):
        state = str(illumination_state or "").upper()
        if state == "PERIODIC_SCAN":
            return "SEARCH"
        if state == "PERSISTENT_ILLUMINATION":
            return "DWELL"
        return "?"

    def update(self, emitter_id, time_s, illumination_state):
        eid = str(emitter_id)
        second = int(float(time_s))
        if self._last_second.get(eid) == second:
            return
        self._last_second[eid] = second
        q = self._history.setdefault(eid, deque(maxlen=self.max_entries))
        q.append(self.label_from_illumination(illumination_state))

    def labels(self, emitter_id):
        return list(self._history.get(str(emitter_id), ()))

    def clear(self):
        self._history.clear()
        self._last_second.clear()
