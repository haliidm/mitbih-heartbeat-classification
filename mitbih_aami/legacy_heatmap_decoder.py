"""Convert dense class predictions into beat events and R-R/heart-rate values."""

import numpy as np

from .label_config import CLASS_NAMES, MAX_LABEL_RADIUS


MIN_REGION_SAMPLES = 8
MAX_REGION_SAMPLES = 2 * MAX_LABEL_RADIUS + 1


def decode_logits(logits: np.ndarray):
    """Return predicted beat events from [classes, samples] logits.

    Each predicted non-background class region represents one beat. Its center
    is used as the R location, matching the centered training target. Regions
    wider than the widest target are recursively split at a low-confidence
    point, which can recover merged adjacent beats.
    """
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] != len(CLASS_NAMES):
        raise ValueError(f"Expected logits with shape [{len(CLASS_NAMES)}, samples].")

    shifted = logits - logits.max(axis=0, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-12)
    labels = logits.argmax(axis=0).astype(np.int64)

    events = []
    for class_id in range(1, len(CLASS_NAMES)):
        indices = np.flatnonzero(labels == class_id)
        if not len(indices):
            continue
        groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        for group in groups:
            pending = [(int(group[0]), int(group[-1]) + 1)]
            while pending:
                start, end = pending.pop()
                length = end - start
                if length > MAX_REGION_SAMPLES:
                    low = max(start + MIN_REGION_SAMPLES, end - MAX_REGION_SAMPLES)
                    high = min(end - MIN_REGION_SAMPLES, start + MAX_REGION_SAMPLES)
                    if high <= low:
                        split_at = (start + end) // 2
                    else:
                        candidate_scores = probabilities[class_id, low:high]
                        split_at = low + int(candidate_scores.argmin())
                    pending.append((split_at, end))
                    pending.append((start, split_at))
                    continue
                if length < MIN_REGION_SAMPLES:
                    continue

                center = (start + end - 1) // 2
                confidence = float(probabilities[class_id, start:end].mean())
                events.append({
                    "sample": center,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": confidence,
                })

    return sorted(events, key=lambda event: event["sample"])


def heart_rate_from_events(events, fs: int = 360):
    """Estimate a robust record heart rate from physiologically plausible RR."""
    samples = np.asarray(sorted(event["sample"] for event in events), dtype=np.int64)
    if len(samples) < 2:
        return None
    rr = np.diff(samples)
    lower = fs * 60.0 / 220.0
    upper = fs * 60.0 / 30.0
    rr = rr[(rr >= lower) & (rr <= upper)]
    if not len(rr):
        return None
    return float(60.0 * fs / np.median(rr))
