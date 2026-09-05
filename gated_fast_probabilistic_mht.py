from fast_probabilistic_mht import (
    FastProbabilisticMultipleHypothesisAssociator,
    _HypothesisState,
    _HistoryNode,
)


class GatedFastProbabilisticMultipleHypothesisAssociator(
    FastProbabilisticMultipleHypothesisAssociator
):
    """Fast PMHT with cheap compatibility gating before probabilistic branching.

    The gate is deliberately wider than the likelihood model. Its purpose is to
    reject obviously incompatible tracks cheaply, not to make the final
    association decision. Only tracks that pass the gate receive the more
    expensive likelihood calculation and branch creation.

    This is still a beam-pruned MHT. Gating reduces the number of branches; it
    does not replace the probabilistic ranking among surviving alternatives.
    """

    def __init__(
        self,
        *args,
        frequency_gate_sigma=4.0,
        pulse_width_gate_sigma=4.0,
        amplitude_gate_sigma=4.0,
        timing_gate_sigma=4.0,
        gate_modulation=True,
        max_track_branches=2,
        birth_when_gated=True,
        clutter_when_gated=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.frequency_gate_sigma = float(frequency_gate_sigma)
        self.pulse_width_gate_sigma = float(pulse_width_gate_sigma)
        self.amplitude_gate_sigma = float(amplitude_gate_sigma)
        self.timing_gate_sigma = float(timing_gate_sigma)
        self.gate_modulation = bool(gate_modulation)
        self.max_track_branches = max(1, int(max_track_branches))
        self.birth_when_gated = bool(birth_when_gated)
        self.clutter_when_gated = bool(clutter_when_gated)

    @staticmethod
    def _within(value, limit):
        return abs(value) <= limit

    def _passes_timing_gate(self, pdw, track):
        pri_s = track.estimated_pri_s
        if pri_s is None:
            return True

        dt_s = pdw.toa_s - track.last_toa_s
        if dt_s <= 0.0:
            return False

        gate_s = self.timing_gate_sigma * self.timing_sigma_s
        best_residual = min(
            abs(dt_s - multiple * pri_s)
            for multiple in range(1, self.max_pri_multiple + 1)
        )
        return best_residual <= gate_s

    def _passes_gate(self, pdw, track):
        if not self._within(
            pdw.frequency_hz - track.mean_frequency_hz,
            self.frequency_gate_sigma * self.frequency_sigma_hz,
        ):
            return False

        if not self._within(
            pdw.pulse_width_s - track.mean_pulse_width_s,
            self.pulse_width_gate_sigma * self.pulse_width_sigma_s,
        ):
            return False

        if not self._within(
            pdw.amplitude_dbfs - track.mean_amplitude_dbfs,
            self.amplitude_gate_sigma * self.amplitude_sigma_db,
        ):
            return False

        if self.gate_modulation:
            if (
                pdw.modulation_type != "UNKNOWN"
                and track.dominant_modulation != "UNKNOWN"
                and pdw.modulation_type != track.dominant_modulation
            ):
                return False

        return self._passes_timing_gate(pdw, track)

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
                compatible = []

                if num_tracks:
                    log_assignment_prior = self._safe_log(p_existing / num_tracks)
                    for index, track in enumerate(state.tracks):
                        if not self._passes_gate(pdw, track):
                            continue
                        increment = (
                            log_assignment_prior
                            + self._track_log_likelihood_fast(pdw, track)
                        )
                        compatible.append((increment, index, track))

                    compatible.sort(key=lambda item: item[0], reverse=True)
                    compatible = compatible[: self.max_track_branches]

                    for increment, index, track in compatible:
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

                # A birth is essential when no existing track is compatible.
                # Optionally retain it even when a compatible track exists so
                # the MHT can still represent a genuinely new emitter close to
                # an existing one.
                allow_birth = (
                    num_tracks < self.max_emitters
                    and (not compatible or self.birth_when_gated)
                )
                if allow_birth:
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

                # Clutter is always retained when nothing gated. In the normal
                # hot path it can be suppressed to avoid an unnecessary branch.
                if not compatible or self.clutter_when_gated:
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
