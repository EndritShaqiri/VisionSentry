from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from distance_estimation.geometry import classify_range_bin
from distance_estimation.types import RangeEstimate, TrackRangeEstimate


@dataclass(slots=True)
class _FilterState:
    mean: float
    variance: float
    predicted_mean_next: float | None = None
    predicted_variance_next: float | None = None


class TrackSmoother:
    def __init__(self, *, process_noise_m2: float, min_measurement_std_m: float, measurement_noise_scale: float) -> None:
        self.process_noise_m2 = max(float(process_noise_m2), 1e-6)
        self.min_measurement_std_m = max(float(min_measurement_std_m), 1e-3)
        self.measurement_noise_scale = max(float(measurement_noise_scale), 1e-6)

    def smooth(self, estimates: list[RangeEstimate], range_bins_m: list[float]) -> list[TrackRangeEstimate]:
        if not estimates:
            return []

        ordered = sorted(estimates, key=lambda item: item.frame_index)
        states: list[_FilterState] = []
        prev_mean = ordered[0].distance_m or 0.0
        prev_var = max((ordered[0].distance_std_m or self.min_measurement_std_m) ** 2, self.min_measurement_std_m**2)

        for idx, estimate in enumerate(ordered):
            dt = 1 if idx == 0 else max(1, estimate.frame_index - ordered[idx - 1].frame_index)
            predicted_mean = prev_mean
            predicted_var = prev_var + (self.process_noise_m2 * dt)
            measurement = estimate.distance_m if estimate.distance_m is not None else predicted_mean
            measurement_std = max(estimate.distance_std_m or self.min_measurement_std_m, self.min_measurement_std_m)
            measurement_var = (measurement_std**2) * self.measurement_noise_scale
            gain = predicted_var / (predicted_var + measurement_var)
            updated_mean = predicted_mean + gain * (measurement - predicted_mean)
            updated_var = max((1.0 - gain) * predicted_var, 1e-6)
            state = _FilterState(mean=updated_mean, variance=updated_var)
            if states:
                states[-1].predicted_mean_next = predicted_mean
                states[-1].predicted_variance_next = predicted_var
            states.append(state)
            prev_mean = updated_mean
            prev_var = updated_var

        smoothed_means = [state.mean for state in states]
        smoothed_vars = [state.variance for state in states]
        for idx in range(len(states) - 2, -1, -1):
            state = states[idx]
            predicted_var_next = state.predicted_variance_next or (state.variance + self.process_noise_m2)
            gain = state.variance / predicted_var_next
            predicted_mean_next = state.predicted_mean_next if state.predicted_mean_next is not None else states[idx + 1].mean
            smoothed_means[idx] = state.mean + gain * (smoothed_means[idx + 1] - predicted_mean_next)
            smoothed_vars[idx] = max(
                state.variance + gain * gain * (smoothed_vars[idx + 1] - predicted_var_next),
                1e-6,
            )

        results = []
        for estimate, filtered_state, refined_mean, refined_var in zip(ordered, states, smoothed_means, smoothed_vars):
            raw_distance = estimate.distance_m if estimate.distance_m is not None else filtered_state.mean
            confidence = estimate.distance_confidence if estimate.distance_confidence is not None else 0.0
            results.append(
                TrackRangeEstimate(
                    frame_index=estimate.frame_index,
                    track_id=estimate.track_id or -1,
                    distance_m_raw=raw_distance,
                    distance_m_filtered=filtered_state.mean,
                    distance_m_refined=refined_mean,
                    distance_std_m=math.sqrt(refined_var),
                    distance_confidence=confidence,
                    range_bin=classify_range_bin(refined_mean, range_bins_m),
                    low_confidence=estimate.low_confidence,
                )
            )
        return results


def smooth_track_ranges(
    estimates: list[RangeEstimate],
    *,
    process_noise_m2: float,
    min_measurement_std_m: float,
    measurement_noise_scale: float,
    range_bins_m: list[float],
) -> dict[int, list[TrackRangeEstimate]]:
    grouped: dict[int, list[RangeEstimate]] = defaultdict(list)
    for estimate in estimates:
        if estimate.track_id is None:
            continue
        grouped[int(estimate.track_id)].append(estimate)

    smoother = TrackSmoother(
        process_noise_m2=process_noise_m2,
        min_measurement_std_m=min_measurement_std_m,
        measurement_noise_scale=measurement_noise_scale,
    )
    return {track_id: smoother.smooth(track_estimates, range_bins_m) for track_id, track_estimates in grouped.items()}
