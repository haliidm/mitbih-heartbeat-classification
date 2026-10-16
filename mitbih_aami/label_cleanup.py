"""Post-process dense class predictions before display and event extraction."""

import numpy as np


MAX_BACKGROUND_GAP_MS = 50.0
MIN_ACTIVE_REGION_MS = 100.0


def _dominant_nonzero_class(region: np.ndarray) -> int:
    values = np.asarray(region, dtype=np.int64)
    values = values[values > 0]
    if values.size == 0:
        return 0
    counts = np.bincount(values)
    return int(np.flatnonzero(counts == counts.max())[0])


def postprocess_pointwise_labels(
    labels: np.ndarray,
    sample_rate: int,
    max_gap_ms: float = MAX_BACKGROUND_GAP_MS,
    min_region_ms: float = MIN_ACTIVE_REGION_MS,
) -> np.ndarray:
    """Fill short same-class background gaps and remove short active regions.

    A zero-valued gap is filled only when the dominant nonzero class on both
    sides agrees. After filling, every continuous nonzero region shorter than
    ``min_region_ms`` is removed, including regions inside the record.
    """
    result = np.asarray(labels).copy()
    if result.ndim != 1:
        raise ValueError("pointwise labels must be a one-dimensional array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if max_gap_ms < 0 or min_region_ms < 0:
        raise ValueError("post-processing durations must be nonnegative")
    if result.size == 0:
        return result

    max_gap_samples = round(max_gap_ms * sample_rate / 1000.0)
    if max_gap_samples > 0:
        active = np.flatnonzero(result != 0)
        if active.size:
            split_points = np.flatnonzero(np.diff(active) > 1) + 1
            runs = np.split(active, split_points)
            for left, right in zip(runs, runs[1:]):
                gap_start = int(left[-1]) + 1
                gap_end = int(right[0])
                if gap_end - gap_start > max_gap_samples:
                    continue
                left_class = _dominant_nonzero_class(result[left[0] : left[-1] + 1])
                right_class = _dominant_nonzero_class(result[right[0] : right[-1] + 1])
                if left_class != 0 and left_class == right_class:
                    result[gap_start:gap_end] = left_class

    min_region_samples = round(min_region_ms * sample_rate / 1000.0)
    if min_region_samples > 0:
        active = np.flatnonzero(result != 0)
        if active.size:
            split_points = np.flatnonzero(np.diff(active) > 1) + 1
            for run in np.split(active, split_points):
                if len(run) < min_region_samples:
                    result[run] = 0

    return result