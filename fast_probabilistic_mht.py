import math
import statistics
from dataclasses import dataclass

from probabilistic_mht import ProbabilisticMultipleHypothesisAssociator


@dataclass(frozen=True, slots=True)
class _HistoryNode:
    pdw_id: int
    previous: object = None


@dataclass(frozen=True, slots=True)
class _TrackState:
    candidate_id: int
    pulse_count: int
    last_pdw_id: int
    last_toa_s: float
    mean_frequency_hz: float
    mean_pulse_width_s: float
    mean_amplitude_dbfs: float
    modulation_counts: tuple
    dominant_modulation: str
    recent_intervals_s: tuple
    estimated_pri_s: object
    history: object


@dataclass(frozen=True, slots=True)
class _HypothesisState:
    log_weight: float
    tracks: tuple
    clutter_history: object = None


class FastProbabilisticMultipleHypothesisAssociator(ProbabilisticMultipleHypothesisAssociator):
    """Lightweight beam-pruned probabilistic MHT.

    This preserves the probabilistic branching model of
    ProbabilisticMultipleHypothesisAssociator, but removes deepcopy and repeated
    full-track rescans from the hot path. Track statistics are updated
    incrementally and PDW membership is retained as a persistent linked history.

    The public output is materialized into the same dictionary structure as the
    reference implementation so existing displays and truth scoring continue to
    work unchanged.
    """

    def __init__(self, *args, pri_history_length=9, **kwargs):
        super().__init__(*args, **kwargs)
        self.pri_history_length = max(3, int(pri_history_length))

    @staticmethod
    def _counts_to_dict(counts):
        return dict(counts)

    @staticmethod
    def _dict_to_counts(counts):
        return tuple(sorted(counts.items()))

    def _new_track(self, candidate_id, pdw):
        counts = ((pdw.modulation_type, 1),)
        return _TrackState(
            candidate_id=candidate_id,
            pulse_count=1,
            last_pdw_id=pdw.pdw_id,
            last_toa_s=pdw.toa_s,
            mean_frequency_hz=pdw.frequency_hz,
            mean_pulse_width_s=pdw.pulse_width_s,
            mean_amplitude_dbfs=pdw.amplitude_dbfs,
            modulation_counts=counts,
            dominant_modulation=pdw.modulation_type,
            recent_intervals_s=(),
            estimated_pri_s=None,
            history=_HistoryNode(pdw.pdw_id),
        )

    def _update_track(self, track, pdw):
        count = track.pulse_count + 1
        inv_count = 1.0 / count

        mean_frequency_hz = track.mean_frequency_hz + (
            pdw.frequency_hz - track.mean_frequency_hz
        ) * inv_count
        mean_pulse_width_s = track.mean_pulse_width_s + (
            pdw.pulse_width_s - track.mean_pulse_width_s
        ) * inv_count
        mean_amplitude_dbfs = track.mean_amplitude_dbfs + (
            pdw.amplitude_dbfs - track.mean_amplitude_dbfs
        ) * inv_count

        counts = self._counts_to_dict(track.modulation_counts)
        counts[pdw.modulation_type] = counts.get(pdw.modulation_type, 0) + 1
        dominant_modulation = max(counts, key=counts.get)

        interval_s = pdw.toa_s - track.last_toa_s
        if interval_s > 0.0:
            recent = track.recent_intervals_s + (interval_s,)
            if len(recent) > self.pri_history_length:
                recent = recent[-self.pri_history_length :]
            estimated_pri_s = statistics.median(recent)
        else:
            recent = track.recent_intervals_s
            estimated_pri_s = track.estimated_pri_s

        return _TrackState(
            candidate_id=track.candidate_id,
            pulse_count=count,
            last_pdw_id=pdw.pdw_id,
            last_toa_s=pdw.toa_s,
            mean_frequency_hz=mean_frequency_hz,
            mean_pulse_width_s=mean_pulse_width_s,
            mean_amplitude_dbfs=mean_amplitude_dbfs,
            modulation_counts=self._dict_to_counts(counts),
            dominant_modulation=dominant_modulation,
            recent_intervals_s=recent,
            estimated_pri_s=estimated_pri_s,
            history=_HistoryNode(pdw.pdw_id, track.history),
        )

    def _modulation_log_likelihood_fast(self, pdw, track):
        if pdw.modulation_type == "UNKNOWN" or track.dominant_modulation == "UNKNOWN":
            return self._safe_log(0.5)
        if pdw.modulation_type == track.dominant_modulation:
            return self._safe_log(self.modulation_match_probability)
        return self._safe_log(1.0 - self.modulation_match_probability)

    def _timing_log_likelihood_fast(self, pdw, track):
        pri_s = track.estimated_pri_s
        if pri_s is None:
            return 0.0

        dt_s = pdw.toa_s - track.last_toa_s
        if dt_s <= 0.0:
            return self._safe_log(1e-12)

        p_miss = min(max(self.missed_pulse_probability, 1e-6), 0.95)
        terms = []
        for multiple in range(1, self.max_pri_multiple + 1):
            missed = multiple - 1
            if multiple < self.max_pri_multiple:
                prior = (1.0 - p_miss) * (p_miss ** missed)
            else:
                prior = p_miss ** missed
            residual_s = dt_s - multiple * pri_s
            terms.append(
                self._safe_log(prior)
                + self._gaussian_log_kernel(residual_s, self.timing_sigma_s)
            )
        return self._logsumexp(terms)

    def _track_log_likelihood_fast(self, pdw, track):
        return (
            self._gaussian_log_kernel(
                pdw.frequency_hz - track.mean_frequency_hz,
                self.frequency_sigma_hz,
            )
            + self._gaussian_log_kernel(
                pdw.pulse_width_s - track.mean_pulse_width_s,
                self.pulse_width_sigma_s,
            )
            + self._gaussian_log_kernel(
                pdw.amplitude_dbfs - track.mean_amplitude_dbfs,
                self.amplitude_sigma_db,
            )
            + self._modulation_log_likelihood_fast(pdw, track)
            + self._timing_log_likelihood_fast(pdw, track)
        )

    @staticmethod
    def _replace_track(tracks, index, replacement):
        return tracks[:index] + (replacement,) + tracks[index + 1 :]

    @staticmethod
    def _history_ids(node):
        ids = []
        while node is not None:
            ids.append(node.pdw_id)
            node = node.previous
        ids.reverse()
        return ids

    def _materialize(self, states, pdw_lookup):
        hypotheses = []
        for state in states:
            candidates = []
            for track in state.tracks:
                pdw_ids = self._history_ids(track.history)
                candidates.append(
                    {
                        "candidate_id": track.candidate_id,
                        "pdws": [pdw_lookup[pdw_id] for pdw_id in pdw_ids],
                        "mean_frequency_hz": track.mean_frequency_hz,
                        "mean_pulse_width_s": track.mean_pulse_width_s,
                        "mean_amplitude_dbfs": track.mean_amplitude_dbfs,
                        "dominant_modulation": track.dominant_modulation,
                        "estimated_pri_s": track.estimated_pri_s,
                    }
                )

            clutter_ids = self._history_ids(state.clutter_history)
            hypotheses.append(
                {
                    "log_weight": state.log_weight,
                    "score": -state.log_weight,
                    "candidates": candidates,
                    "clutter_pdws": [pdw_lookup[pdw_id] for pdw_id in clutter_ids],
                    "history": [],
                }
            )

        self._normalize_weights(hypotheses)
        return hypotheses

    def associate(self, pdws):
        if not pdws:
            return []

        pdw_lookup = {pdw.pdw_id: pdw for pdw in pdws}
        states = [_HypothesisState(log_weight=0.0, tracks=())]

        p_birth = min(max(self.birth_probability, 1e-9), 0.95)
        p_clutter = min(max(self.clutter_probability, 1e-9), 0.95)
        p_existing = max(1.0 - p_birth - p_clutter, 1e-9)
        log_birth = self._safe_log(p_birth)
        log_clutter = self._safe_log(p_clutter)

        for pdw in pdws:
            branches = []

            for state in states:
                num_tracks = len(state.tracks)

                if num_tracks:
                    log_assignment_prior = self._safe_log(p_existing / num_tracks)
                    for index, track in enumerate(state.tracks):
                        increment = (
                            log_assignment_prior
                            + self._track_log_likelihood_fast(pdw, track)
                        )
                        updated_track = self._update_track(track, pdw)
                        branches.append(
                            _HypothesisState(
                                log_weight=state.log_weight + increment,
                                tracks=self._replace_track(
                                    state.tracks, index, updated_track
                                ),
                                clutter_history=state.clutter_history,
                            )
                        )

                if num_tracks < self.max_emitters:
                    candidate_id = num_tracks + 1
                    branches.append(
                        _HypothesisState(
                            log_weight=state.log_weight + log_birth,
                            tracks=state.tracks + (
                                self._new_track(candidate_id, pdw),
                            ),
                            clutter_history=state.clutter_history,
                        )
                    )

                branches.append(
                    _HypothesisState(
                        log_weight=state.log_weight + log_clutter,
                        tracks=state.tracks,
                        clutter_history=_HistoryNode(
                            pdw.pdw_id, state.clutter_history
                        ),
                    )
                )

            branches.sort(key=lambda item: item.log_weight, reverse=True)
            states = branches[: self.beam_width]

        return self._materialize(states, pdw_lookup)
