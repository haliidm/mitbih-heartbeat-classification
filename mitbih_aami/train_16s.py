"""Train from scratch on MIT-BIH only, with configurable S-centered sampling."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .label_config import (
    CLASS_NAMES,
    FS,
    LABEL_RADIUS_MS,
    MAX_LABEL_RADIUS,
    NUM_CLASSES,
    PEAK_REFINEMENT_RADIUS_MS,
    SPLIT_RATIOS,
    WINDOW_HALF,
    WINDOW_SAMPLES,
)
from .ecg_data import EventCenteredWindowDataset, available_record_names, load_records
from .cnn_transformer import MultiScaleCNNTransformer
from .sample_rhythm_windows import add_rhythm_n_windows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("mit-bih-arrhythmia-database-1.0.0"),
        help="Folder containing MIT-BIH .hea/.dat/.atr records.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/mitbih_rhythm_n")
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s-sampling-factor", type=float, default=2.0,
                        help="S中心训练窗口相对抽样权重；1保持原始shuffle，默认2")
    parser.add_argument("--rhythm-n-per-record", type=int, default=6, help="每条记录每种AFIB/AFL最多补充的N窗口数，默认6")
    parser.add_argument("--prepare-only", action="store_true", help="仅统计窗口来源及划分，不训练")
    args = parser.parse_args()
    if args.rhythm_n_per_record < 1:
        parser.error("rhythm-n-per-record必须>=1")
    if min(args.epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs、batch-size、patience必须为正整数")
    if not np.isfinite([args.s_sampling_factor, args.learning_rate, args.weight_decay]).all() or args.s_sampling_factor < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("S抽样权重须>=1，学习率须>0，weight-decay须>=0，且均为有限数")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_window_split(dataset_size: int, seed: int):
    if dataset_size < 3:
        raise ValueError(f"Need at least 3 windows for a train/val/test split; got {dataset_size}.")
    permutation = np.random.default_rng(seed).permutation(dataset_size)
    train_end = int(dataset_size * SPLIT_RATIOS[0])
    val_end = int(dataset_size * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    train_indices = permutation[:train_end].tolist()
    val_indices = permutation[train_end:val_end].tolist()
    test_indices = permutation[val_end:].tolist()
    return train_indices, val_indices, test_indices


def make_class_weights(dataset, train_indices, device):
    counts = dataset.pointwise_support(train_indices).astype(np.float64)
    positive = counts[1:]
    present = positive > 0
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    if present.any():
        total = positive[present].sum()
        raw = np.sqrt(total / (present.sum() * positive[present]))
        raw = np.clip(raw, 0.5, 5.0)
        raw = raw / raw.mean()
        for class_id, weight in zip(np.flatnonzero(present) + 1, raw):
            weights[class_id] = float(weight)
    weights[0] = 0.15
    print("Training support (point samples / centered beats):")
    center_counts = dataset.center_class_support(train_indices)
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id} {class_name}: {int(counts[class_id]):,} point samples / "
            f"{int(center_counts[class_id]):,} center beats"
        )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0.0
    total_valid = 0
    for x, target, mask, _, _, _ in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        per_sample_loss = F.cross_entropy(
            logits, target, weight=class_weights, reduction="none"
        )
        loss = (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        valid_count = int(mask.sum().item())
        total_loss += float(loss.item()) * valid_count
        total_valid += valid_count
    return total_loss / max(total_valid, 1)


def confusion_report(confusion, class_ids):
    per_class = {}
    supported_f1 = []
    for class_id in class_ids:
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        support = int(confusion[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class[CLASS_NAMES[class_id]] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support > 0:
            supported_f1.append(f1)
    return {
        "per_class": per_class,
        "macro_f1_supported_classes": (
            float(np.mean(supported_f1)) if supported_f1 else 0.0
        ),
    }


@torch.inference_mode()
def evaluate_pointwise_windows(model, dataset, indices, device, batch_size):
    """Evaluate pointwise class labels only; no beat-event matching is used."""
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, target, mask, _, _, _ in loader:
        x = x.to(device, non_blocking=True)
        predicted = model(x).argmax(dim=1).cpu().numpy()
        target_np = target.numpy()
        valid = mask.numpy() > 0
        encoded = target_np[valid].astype(np.int64) * NUM_CLASSES + predicted[valid]
        confusion += np.bincount(
            encoded, minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    all_classes = confusion_report(confusion, range(NUM_CLASSES))
    beat_classes = confusion_report(confusion, range(1, NUM_CLASSES))
    return {
        "per_class": all_classes["per_class"],
        "macro_f1_all_classes": all_classes["macro_f1_supported_classes"],
        "macro_f1_N_S_V_F_Q": beat_classes["macro_f1_supported_classes"],
        "point_count": int(confusion.sum()),
        "confusion_matrix": confusion.tolist(),
    }


def _split_manifest(dataset, split_indices, seed, record_names):
    output = {
        "seed": seed,
        "ratios": {
            "train": SPLIT_RATIOS[0],
            "validation": SPLIT_RATIOS[1],
            "test": SPLIT_RATIOS[2],
        },
        "split_unit": "event-centered window",
        "record_names": list(record_names),
    }
    for split_name, indices in split_indices.items():
        output[split_name] = []
        for index in indices:
            record_index, center_sample, center_class, original_sample = dataset.items[index]
            output[split_name].append({
                "record": dataset.records[record_index].name,
                "lead": dataset.records[record_index].lead_name,
                "center_sample": center_sample,
                "original_atr_sample": original_sample,
                "refinement_offset_ms": round((center_sample - original_sample) * 1000.0 / FS, 3),
                "center_class": CLASS_NAMES[center_class],
            })
    output["ratios_apply_to"] = "original S/V/F/Q windows; supplemental split details in rhythm_n_windows.json"
    return output


def write_refinement_audit(records, output_path):
    fields = [
        "record", "lead", "class_id", "class_name", "original_atr_sample",
        "corrected_peak_sample", "offset_samples", "offset_ms", "polarity",
        "amplitude_z", "sharpness", "search_left", "search_right_exclusive",
        "search_radius_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for item in record.refinement_info:
                writer.writerow({
                    "record": record.name,
                    "lead": record.lead_name,
                    "class_id": item["class_id"],
                    "class_name": CLASS_NAMES[item["class_id"]],
                    "original_atr_sample": item["original_sample"],
                    "corrected_peak_sample": item["corrected_sample"],
                    "offset_samples": item["offset_samples"],
                    "offset_ms": f'{item["offset_samples"] * 1000.0 / FS:.3f}',
                    "polarity": item["polarity"],
                    "amplitude_z": f'{item["amplitude_z"]:.6f}',
                    "sharpness": f'{item["sharpness"]:.6f}',
                    "search_left": item["search_left"],
                    "search_right_exclusive": item["search_right_exclusive"],
                    "search_radius_ms": item["search_radius_ms"],
                })


def write_pointwise_labels(records, output_dir):
    """Persist regenerated pointwise labels and corrected/raw event locations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        np.savez_compressed(
            output_dir / f"{record.name}_labels.npz",
            target=record.target,
            corrected_samples=np.asarray(
                [sample for sample, _ in record.events], dtype=np.int64
            ),
            class_ids=np.asarray(
                [class_id for _, class_id in record.events], dtype=np.uint8
            ),
            original_atr_samples=np.asarray(
                [sample for sample, _ in record.original_events], dtype=np.int64
            ),
        )



def training_sampler(dataset, train_indices, factor, seed):
    """Sample only the training subset; preserve original shuffle at factor=1."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("S sampling factor must be finite and >=1")
    if factor == 1:
        return None
    weights = [factor if dataset.items[index][2] == 2 else 1.0 for index in train_indices]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(train_indices),
                                 replacement=True, generator=generator)


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("输出目录非空，请使用新的--output-dir，避免覆盖已有实验。")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pointwise label region: about {LABEL_RADIUS_MS * 2} ms total "
          f"(±{LABEL_RADIUS_MS} ms; {MAX_LABEL_RADIUS} samples each side)")
    record_names = available_record_names(args.database_dir)
    expected = {"100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
                "111", "112", "113", "114", "115", "116", "117", "118", "119",
                "121", "122", "123", "124", "200", "201", "202", "203", "205",
                "207", "208", "209", "210", "212", "213", "214", "215", "217",
                "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"}
    if set(record_names) != expected:
        raise ValueError("此入口要求完整的48条MIT-BIH记录，不接受ST-T数据库。")
    print(f"Loading MIT-BIH records: {len(record_names)}; random initialization; no pretrained weights")

    records = load_records(args.database_dir, record_names)
    dataset = EventCenteredWindowDataset(records)
    if not len(dataset):
        raise RuntimeError("No S/V/F/Q-centered windows were created.")

    train_indices, val_indices, test_indices = random_window_split(
        len(dataset), args.seed
    )
    splits = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
    }
    rhythm_audit = add_rhythm_n_windows(dataset, splits, args.database_dir,
                                       cap=args.rhythm_n_per_record, seed=args.seed)
    print(f"Added AFIB/AFL N windows: {rhythm_audit['supplemental_window_count']}")
    for row in rhythm_audit["records"]:
        print(f"  {row['record']} {row['rhythm']}: eligible={row['eligible']}, selected={row['selected']}")
    print(
        f"Windows: total={len(dataset)}, train={len(train_indices)}, "
        f"validation={len(val_indices)}, test={len(test_indices)} "
        "(original windows retain 7:1:2 split; disjoint rhythm N windows added)"
    )
    for split_name, indices in splits.items():
        counts = dataset.center_class_support(indices)
        summary = ", ".join(
            f"{CLASS_NAMES[class_id]}={int(counts[class_id])}"
            for class_id in range(1, NUM_CLASSES)
        )
        print(f"  {split_name} center beats: {summary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rhythm_n_windows.json").write_text(json.dumps(rhythm_audit,indent=2,ensure_ascii=False),encoding="utf-8")
    if args.prepare_only:
        (args.output_dir / "split_manifest.json").write_text(
            json.dumps(_split_manifest(dataset,splits,args.seed,record_names),indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"Prepared only: {args.output_dir.resolve()}; no training started.")
        return
    write_refinement_audit(records, args.output_dir / 'annotation_refinement.csv')
    write_pointwise_labels(records, args.output_dir / 'pointwise_labels')
    with (args.output_dir / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            _split_manifest(dataset, splits, args.seed, record_names),
            file,
            indent=2,
            ensure_ascii=False,
        )

    rhythm_val_indices = [row["dataset_index"] for row in rhythm_audit["windows"] if row["split"] == "validation"]
    sampling = training_sampler(dataset, train_indices, args.s_sampling_factor, args.seed)
    center_counts = dataset.center_class_support(train_indices)
    s_count = int(center_counts[2])
    denominator = len(train_indices) + (args.s_sampling_factor - 1) * s_count
    expected_s_fraction = args.s_sampling_factor * s_count / denominator
    print(f"S-center sampling factor={args.s_sampling_factor:g}; expected S-center share={expected_s_fraction:.2%}")
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"initialization": "random", "training_database": "MIT-BIH only",
                   "external_database_used": False,
                   "rhythm_N_supplemental_count": rhythm_audit["supplemental_window_count"],
                   "rhythm_sampling": "uniform AFIB/AFL N-only windows outside all original windows", "selection": "MIT-BIH validation pointwise macro-F1 N/S/V/F/Q",
                   "sampling": "S-weighted replacement" if sampling is not None else "original shuffle",
                   "windows_per_epoch": len(train_indices), "expected_S_center_fraction": expected_s_fraction,
                   "split_unit": "window; records and overlapping windows may cross internal splits"})
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=sampling is None,
        sampler=sampling,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = MultiScaleCNNTransformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    class_weights = make_class_weights(dataset, train_indices, device)
    config["class_weights"] = class_weights.detach().cpu().tolist()
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, class_weights, device
        )
        validation = evaluate_pointwise_windows(
            model, dataset, val_indices, device, args.batch_size
        )
        rhythm_validation = evaluate_pointwise_windows(
            model, dataset, rhythm_val_indices, device, args.batch_size
        ) if rhythm_val_indices else None
        if rhythm_validation:
            cm = np.asarray(rhythm_validation["confusion_matrix"])
            n_to_s = float(cm[1,2] / max(1,cm[1].sum()))
            rhythm_validation["reference_N_predicted_S_fraction"] = n_to_s
            print(f"  AFIB/AFL N validation: N->S point fraction={n_to_s:.4%}")
        score = validation["macro_f1_N_S_V_F_Q"]
        scheduler.step(score)
        learning_rate = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_pointwise_macro_f1_N_S_V_F_Q": score,
            "learning_rate": learning_rate,
            "rhythm_N_validation": rhythm_validation,
            "validation_per_class": validation["per_class"],
            "validation_confusion_matrix": validation["confusion_matrix"],
        })
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={train_loss:.5f}, "
            f"validation pointwise macro-F1 (N/S/V/F/Q)={score:.4f}, "
            f"lr={learning_rate:.2e}"
        )
        print(
            "  Per-class pointwise F1: "
            + ", ".join(
                f"{name}={metrics['f1']:.3f}"
                for name, metrics in validation["per_class"].items()
                if name != "background"
            )
        )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_pointwise_macro_f1_N_S_V_F_Q": score,
                "class_names": CLASS_NAMES,
                "record_names": record_names,
                "training_config": config,
                "split_seed": args.seed,
                "split_ratios": SPLIT_RATIOS,
                "peak_refinement_radius_ms": PEAK_REFINEMENT_RADIUS_MS,
                "label_radius_ms": LABEL_RADIUS_MS,
                "label_reference": "corrected positive/negative local peak near original ATR",
            }, args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        with (args.output_dir / "training_history.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2, ensure_ascii=False)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was selected.")

    print(f"Best epoch: {best_epoch}; validation pointwise macro-F1 (N/S/V/F/Q): {best_score:.4f}")
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_pointwise = evaluate_pointwise_windows(
        model, dataset, test_indices, device, args.batch_size
    )
    test_metrics = {
        "best_epoch": best_epoch,
        "label_width_ms": LABEL_RADIUS_MS * 2,
        "pointwise_classification": test_pointwise,
        "split_seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "record_names": record_names,
    }
    with (args.output_dir / "pointwise_classification_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(test_metrics, file, indent=2, ensure_ascii=False)
    print("Held-out pointwise classification:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
