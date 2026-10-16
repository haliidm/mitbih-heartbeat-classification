"""Run a trained model on one MIT-BIH record and export detected beats."""

import argparse
import csv
from pathlib import Path

import torch

from .label_config import FS
from .ecg_data import load_signal_only
from .legacy_heatmap_decoder import decode_logits
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, default=Path("mit-bih-arrhythmia-database-1.0.0"))
    parser.add_argument("--record", required=True, help="MIT-BIH record ID, for example 100.")
    parser.add_argument("--output", type=Path, default=Path("runs/predicted_beats.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MultiScaleCNNTransformer().to(device)
    model.load_state_dict(checkpoint["model_state"])

    record = load_signal_only(args.database_dir, args.record)
    logits = predict_record(model, record, device)
    events = decode_logits(logits)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "sample", "time_seconds", "class_id", "class_name", "confidence",
            "rr_samples", "rr_ms", "instantaneous_hr_bpm",
        ])
        writer.writeheader()
        previous_sample = None
        for event in events:
            rr = event["sample"] - previous_sample if previous_sample is not None else None
            writer.writerow({
                "sample": event["sample"],
                "time_seconds": f"{event['sample'] / FS:.6f}",
                "class_id": event["class_id"],
                "class_name": event["class_name"],
                "confidence": f"{event['confidence']:.6f}",
                "rr_samples": rr if rr is not None else "",
                "rr_ms": f"{rr * 1000.0 / FS:.2f}" if rr is not None else "",
                "instantaneous_hr_bpm": (
                    f"{60.0 * FS / rr:.2f}" if rr and rr > 0 else ""
                ),
            })
            previous_sample = event["sample"]

    print(f"Detected {len(events)} beats.")
    print(f"Predictions saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
