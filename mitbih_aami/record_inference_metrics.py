"""Window event metrics and full-record inference stitching."""

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .label_config import (
    FS,
    INFERENCE_STRIDE,
    NUM_CLASSES,
    PAD_SAMPLES,
    WINDOW_SAMPLES,
)


def _match_one_class(
    truth_samples: Sequence[int],
    pred_samples: Sequence[int],
    tolerance_samples: int,
) -> Tuple[int, int, int]:
    truth = sorted(int(sample) for sample in truth_samples)
    predicted = sorted(int(sample) for sample in pred_samples)
    i = j = true_positive = 0
    while i < len(truth) and j < len(predicted):
        difference = predicted[j] - truth[i]
        if difference < -tolerance_samples:
            j += 1
        elif difference > tolerance_samples:
            i += 1
        else:
            true_positive += 1
            i += 1
            j += 1
    return true_positive, len(predicted) - true_positive, len(truth) - true_positive


def _safe_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "recall": recall, "f1": f1,
    }


def event_metrics(
    truths: Sequence[Sequence[Tuple[int, int]]],
    predictions: Sequence[Sequence[dict]],
    tolerance_ms: int,
) -> Dict:
    """Match reference and predicted events independently inside each window."""
    tolerance_samples = round(tolerance_ms * FS / 1000.0)
    per_class = {}
    total_tp = total_fp = total_fn = 0
    supported_f1 = []

    for class_id in range(1, NUM_CLASSES):
        tp = fp = fn = class_truth_count = 0
        for window_truth, window_predictions in zip(truths, predictions):
            class_truth = [
                sample for sample, label in window_truth if label == class_id
            ]
            class_pred = [
                event["sample"] for event in window_predictions
                if event["class_id"] == class_id
            ]
            matched = _match_one_class(class_truth, class_pred, tolerance_samples)
            tp += matched[0]
            fp += matched[1]
            fn += matched[2]
            class_truth_count += len(class_truth)

        score = _safe_prf(tp, fp, fn)
        per_class[str(class_id)] = score
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if class_truth_count:
            supported_f1.append(score["f1"])

    aggregate = _safe_prf(total_tp, total_fp, total_fn)
    aggregate["macro_f1_supported_classes"] = (
        float(np.mean(supported_f1)) if supported_f1 else 0.0
    )
    aggregate["tolerance_ms"] = tolerance_ms
    aggregate["per_class"] = per_class
    return aggregate


@torch.inference_mode()
def predict_record(model, record, device, batch_windows: int = None) -> np.ndarray:
    """Predict and overlap-average logits for one full record."""
    model.eval()
    window_samples = int(getattr(model, "input_samples", WINDOW_SAMPLES))
    if batch_windows is None:
        batch_windows = 16 if window_samples <= WINDOW_SAMPLES else 2 if window_samples <= 32 * FS else 1
    half_window = window_samples // 2
    starts = list(range(0, len(record.signal), INFERENCE_STRIDE))
    summed = np.zeros((NUM_CLASSES, len(record.signal)), dtype=np.float32)
    counts = np.zeros(len(record.signal), dtype=np.float32)
    # Align the model output sample 0 with the current stitch position. Reflect
    # padding supplies context at the record end, and the left pad keeps indexing simple.
    padded_signal = np.pad(record.signal, (half_window, window_samples), mode="reflect")

    for offset in range(0, len(starts), batch_windows):
        batch_starts = starts[offset:offset + batch_windows]
        batch = np.stack([
            padded_signal[half_window + start:half_window + start + window_samples]
            for start in batch_starts
        ])
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        logits = model(x).cpu().numpy()
        for local_index, start in enumerate(batch_starts):
            valid = min(window_samples, len(record.signal) - start)
            summed[:, start:start + valid] += logits[local_index, :, :valid]
            counts[start:start + valid] += 1.0

    return summed / np.maximum(counts[None, :], 1.0)
