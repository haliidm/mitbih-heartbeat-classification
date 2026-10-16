"""Fine-tune the existing classifier on patient-grouped European ST-T data."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import wfdb

from .label_config import CLASS_NAMES, FS, LABEL_RADIUS_MS, NUM_CLASSES, SYMBOL_TO_CLASS
from .ecg_data import EventCenteredWindowDataset, available_record_names, load_record_for_inference
from .legacy_evaluate_records_v4 import count_record_events, calculate_metrics, extract_predicted_beats, match_by_position
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer
from .legacy_beat_postprocess_v4 import RULE_DESCRIPTION
from .legacy_train_baseline import set_seed, train_one_epoch, write_refinement_audit


# Explicit same-subject relations published by PhysioNet. Apply these
# relations to records actually present; the distribution has no e0120.
GROUP_SOURCE = "https://physionet.org/content/edb/1.0.0/"
SAME_PATIENT = (
    ("e0118", "e0119", "e0120", "e0121", "e0122"),
    ("e0123", "e0124", "e0125", "e0126"),
    ("e0129", "e0133"), ("e0136", "e0139"), ("e0147", "e0148"),
    ("e0154", "e0155"), ("e0162", "e0163"),
)
SPLITS = ("train", "validation", "test")


def patient_group(name):
    return next((group[0] for group in SAME_PATIENT if name in group), name)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def annotation_counts(database_dir, names):
    rows = {}
    for name in names:
        header = wfdb.rdheader(str(database_dir / name))
        annotation = wfdb.rdann(str(database_dir / name), "atr")
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            category = SYMBOL_TO_CLASS.get(symbol)
            if category is not None and 0 <= sample < header.sig_len:
                counts[category] += 1
        rows[name] = counts
    return rows


def create_split(names, counts, seed):
    """Choose a reproducible group split using annotation counts, never predictions."""
    groups = sorted({patient_group(name) for name in names})
    if len(groups) < 5:
        raise ValueError("至少需要5个患者组才能建立训练、验证、测试划分。")
    by_group = {group: [name for name in names if patient_group(name) == group] for group in groups}
    matrix = np.array([sum((counts[name] for name in by_group[group]), np.zeros(NUM_CLASSES, dtype=np.int64)) for group in groups])[:, 1:]
    total = matrix.sum(axis=0)
    support_groups = (matrix > 0).sum(axis=0)
    ratios = np.array([0.6, 0.2, 0.2])
    n_val = n_test = max(1, round(len(groups) * 0.2))
    n_train = len(groups) - n_val - n_test
    rng = np.random.default_rng(seed)
    best_score, best = float("inf"), None
    weights = np.array([1., 2., 2., 2., 0.1])
    for _ in range(5000):
        order = rng.permutation(len(groups))
        parts = (order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:])
        sums = np.array([matrix[part].sum(axis=0) for part in parts])
        fraction = sums / np.maximum(total, 1)
        score = float((((fraction - ratios[:, None]) ** 2) * weights * (total > 0)).sum())
        # Favor training coverage; seek coverage in all splits where >=3 groups carry a class.
        score += 20 * int(((sums[0] == 0) & (total > 0)).sum())
        score += 10 * int(((sums == 0) & (support_groups >= 3)[None, :]).sum())
        if score < best_score:
            best_score, best = score, parts
    result = {key: sorted(name for i in part for name in by_group[groups[int(i)]])
              for key, part in zip(SPLITS, best)}
    return {"schema_version": 1, "seed": seed, "split_unit": "patient_group",
            "group_source": GROUP_SOURCE, "target_group_ratios": ratios.tolist(),
            "patient_groups": {name: patient_group(name) for name in names},
            **result}


def validate_split(manifest, names):
    seen_records, seen_groups = set(), set()
    for split in SPLITS:
        records = manifest.get(split)
        if not isinstance(records, list) or not records or len(records) != len(set(records)):
            raise ValueError(f"{split}必须是非空、不重复的记录名列表。")
        if seen_records.intersection(records):
            raise ValueError("同一记录出现在不同集合中。")
        groups = {patient_group(name) for name in records}
        if seen_groups.intersection(groups):
            raise ValueError("同一患者的记录被分到了不同集合。")
        seen_groups.update(groups)
        seen_records.update(records)
    if seen_records != set(names):
        raise ValueError("划分中的记录必须与数据库的全部记录一一对应。")


def prepare(args):
    names = available_record_names(args.database_dir)
    if any(not (len(name) == 5 and name.startswith("e") and name[1:].isdigit()) for name in names):
        raise ValueError("此入口仅针对European ST-T记录命名；其他数据库需要独立患者映射。")
    counts = annotation_counts(args.database_dir, names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.split_file or args.output_dir / "split_manifest.json"
    if source.exists():
        manifest = json.loads(source.read_text(encoding="utf-8-sig"))
    elif args.split_file:
        raise FileNotFoundError(source)
    else:
        manifest = create_split(names, counts, args.seed)
    validate_split(manifest, names)
    manifest["class_counts"] = {
        split: {CLASS_NAMES[i]: int(sum(counts[name][i] for name in manifest[split]))
                for i in range(1, NUM_CLASSES)} for split in SPLITS}
    existing = args.output_dir / "split_manifest.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if any(old[key] != manifest[key] for key in SPLITS):
            raise ValueError("输出目录已保存不同划分，请使用新输出目录。")
    write_json(existing, manifest)
    with (args.output_dir / "record_class_counts.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["记录", "患者组", "集合", *CLASS_NAMES[1:]])
        for split in SPLITS:
            for name in manifest[split]:
                writer.writerow([name, patient_group(name), split, *counts[name][1:].tolist()])
    for split in SPLITS:
        print(f"{split}: {len(manifest[split])}条记录, "
              f"{len({patient_group(name) for name in manifest[split]})}患者组; {manifest['class_counts'][split]}")
        missing = [name for name, count in manifest["class_counts"][split].items() if count == 0]
        if missing:
            print(f"  本集合缺少参考类别：{', '.join(missing)}；不宣称这些类别已获充分评估。")
    return manifest


class FineTuneWindows(EventCenteredWindowDataset):
    """Only S/V/F/Q-centered 16s windows; retain all pointwise labels inside."""
    def __init__(self, records, minority_repeats=1):
        if minority_repeats < 1:
            raise ValueError("minority_repeats must be at least 1")
        super().__init__(records)
        self.items = self.items * minority_repeats
        if not self.items:
            raise ValueError("训练记录中没有S/V/F/Q中心心搏，无法生成训练窗口。")
        self.natural_count = 0
        self.centered_count = len(self.items)


def validation_full_records(model, args, names, device):
    """Full-record inference; no oversampling, no repeated-window metric counting."""
    totals = None
    matched_matrix = np.zeros((5, 5), dtype=np.int64)
    point_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for name in names:
        print(f"验证完整记录 {name} ...", flush=True)
        record, _ = load_record_for_inference(args.database_dir, name)
        logits = predict_record(model, record, device, batch_windows=args.batch_size)
        labels = logits.argmax(axis=0).astype(np.uint8)
        predicted = extract_predicted_beats(labels)
        counts, _, _, _ = count_record_events(record.events, predicted, round(.1 * FS))
        if totals is None:
            totals = counts
        else:
            for category in totals:
                for key in totals[category]:
                    totals[category][key] += counts[category][key]
        matches, _, _ = match_by_position(record.events, predicted, round(.1 * FS))
        for (_, truth), (_, prediction) in matches:
            matched_matrix[truth-1, prediction-1] += 1
        encoded = record.target.astype(np.int64) * NUM_CLASSES + labels
        point_matrix += np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        del record, logits, labels, encoded
    metrics = calculate_metrics(totals)
    for category, item in metrics.items():
        denominator = item["TP"] + item["FP"]
        item["precision"] = item["TP"] / denominator if denominator else None
    supported = [name for name in args.selection_classes if metrics[name]["TP"] + metrics[name]["FN"] > 0]
    if not supported:
        raise ValueError("验证集不包含任何选模目标类别。")
    score = float(np.mean([metrics[name]["F1分数"] for name in supported]))
    return {"selection_macro_f1": score, "selection_supported_classes": supported,
            "metrics": metrics, "matched_confusion_class_order": list(CLASS_NAMES[1:]),
            "matched_confusion_matrix": matched_matrix.tolist(),
            "pointwise_confusion_matrix": point_matrix.tolist(), "v4_rules": RULE_DESCRIPTION}


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("label_radius_ms") != LABEL_RADIUS_MS:
        raise ValueError("初始模型需包含label_radius_ms且为180毫秒。")
    if tuple(checkpoint.get("class_names", CLASS_NAMES)) != tuple(CLASS_NAMES):
        raise ValueError("初始模型类别顺序不兼容。")
    model = MultiScaleCNNTransformer().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint.get("epoch")


def evaluate_test(args, manifest):
    config_path = args.output_dir / "finetune_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(config["database_dir"]).resolve() != args.database_dir.resolve():
        raise ValueError("测试数据库目录与微调记录不一致。")
    models = (("test_pretrained_v4", Path(config["checkpoint"])),
              ("test_finetuned_v4", args.output_dir / "best_model.pt"))
    for _, path in models:
        if not path.is_file():
            raise FileNotFoundError(path)
    for folder, checkpoint in models:
        subprocess.run([sys.executable, "-m", "mitbih_aami.legacy_evaluate_records_v4",
                        "--database-dir", str(args.database_dir.resolve()),
                        "--checkpoint", str(checkpoint.resolve()),
                        "--output-dir", str((args.output_dir / folder).resolve()),
                        "--batch-windows", str(args.batch_size), "--tolerance-ms", "100",
                        "--records", *manifest["test"]],
                       cwd=str(Path(__file__).resolve().parent.parent), check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="European ST-T患者分组微调入口")
    parser.add_argument("--database-dir", type=Path, default=Path("european-st-t-database-1.0.0"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mitbih_rhythm_n/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stt_finetune"))
    parser.add_argument("--split-file", type=Path, help="已有患者分组JSON；自动校验记录/患者无交叉")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="仅统计类别并保存患者划分，不训练")
    mode.add_argument("--evaluate-test", action="store_true", help="仅对保留测试记录评估微调前后两个模型")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.15)
    parser.add_argument("--minority-repeats", type=int, default=1, help="S/V/F/Q中心窗口重复次数，默认1，至少1；不添加自然窗口")
    parser.add_argument("--selection-classes", nargs="+", choices=list(CLASS_NAMES[1:]), default=["N", "S", "V", "F"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.minority_repeats < 1:
        parser.error("轮数、早停轮数、batch-size及minority-repeats均须为正整数。")
    if not np.isfinite([args.learning_rate, args.weight_decay, args.background_weight]).all() or args.learning_rate <= 0 or args.weight_decay < 0 or args.background_weight <= 0:
        parser.error("学习率和背景权重须为有限正数，weight-decay须非负。")
    args.selection_classes = list(dict.fromkeys(args.selection_classes))
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.prepare_only and not args.evaluate_test:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        if any((args.output_dir / name).exists() for name in ("best_model.pt", "finetune_config.json")):
            raise FileExistsError("该目录已有微调运行。请换用新的--output-dir，避免覆盖；测试使用--evaluate-test。")
    manifest = prepare(args)
    if args.prepare_only:
        print(f"仅完成划分和统计：{args.output_dir.resolve()}")
        return
    if args.evaluate_test:
        evaluate_test(args, manifest)
        return
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前PyTorch未检测到可用CUDA。")
    device = torch.device(device_name)
    print(f"Device: {device}; 初始化权重：{args.checkpoint}")
    if sum(manifest["class_counts"]["train"][name] for name in ("S", "V", "F", "Q")) == 0:
        raise ValueError("训练集合没有S/V/F/Q心搏，请检查患者划分。")
    model, pretrained_epoch = load_model(args.checkpoint, device)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"sampling_rate": FS, "label_radius_ms": LABEL_RADIUS_MS,
                   "split_source": str((args.output_dir / 'split_manifest.json').resolve()),
                   "optimizer": "fresh AdamW", "loss": "masked full-window cross entropy",
                   "training_sampling": "S/V/F/Q-centered windows only; all pointwise classes retained",
                   "v4_rules": RULE_DESCRIPTION, "test_used_during_training": False})
    write_json(args.output_dir / "finetune_config.json", config)
    history = []

    def save_best(epoch, validation):
        torch.save({"model_state": model.state_dict(), "epoch": epoch,
                    "pretrained_epoch": pretrained_epoch, "class_names": CLASS_NAMES,
                    "label_radius_ms": LABEL_RADIUS_MS, "sample_rate": FS,
                    "source_checkpoint": str(args.checkpoint.resolve()),
                    "record_names": manifest["train"], "patient_split": manifest,
                    "validation": validation}, args.output_dir / "best_model.pt")

    print("先评估初始模型在验证集上的表现（第0轮）。", flush=True)
    baseline = validation_full_records(model, args, manifest["validation"], device)
    write_json(args.output_dir / "baseline_validation.json", baseline)
    best_score, best_epoch, stale = baseline["selection_macro_f1"], 0, 0
    save_best(0, baseline)
    history.append({"epoch": 0, "validation": baseline})
    write_json(args.output_dir / "training_history.json", history)
    records = [load_record_for_inference(args.database_dir, name)[0] for name in manifest["train"]]
    dataset = FineTuneWindows(records, args.minority_repeats)
    write_refinement_audit(records, args.output_dir / "training_annotation_refinement.csv")
    write_json(args.output_dir / "training_windows.json", {
        "natural_count": 0, "minority_centered_count": dataset.centered_count,
        "windows": [{"record": records[i].name, "center_sample": center,
                     "center_class": category, "kind": "minority_centered"}
                    for i, center, category, _ in dataset.items]})
    print(f"训练窗口：{dataset.centered_count}个S/V/F/Q中心窗口；不添加自然窗口；窗口内背景和N/S/V/F/Q全部有效点参与损失。")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    weights = torch.ones(NUM_CLASSES, device=device)
    weights[0] = args.background_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=.5)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, weights, device)
        validation = validation_full_records(model, args, manifest["validation"], device)
        score = validation["selection_macro_f1"]
        history.append({"epoch": epoch, "loss": loss, "learning_rate": optimizer.param_groups[0]["lr"], "validation": validation})
        write_json(args.output_dir / "training_history.json", history)
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.5f}, 验证分类macro-F1={score:.4f}, "
              f"验证R峰F1={validation['metrics']['R峰']['F1分数']}", flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            save_best(epoch, validation)
        else:
            stale += 1
        scheduler.step(score)
        if stale >= args.patience:
            print(f"验证指标连续{args.patience}轮未改善，早停。")
            break
    write_json(args.output_dir / "finetune_summary.json", {"best_epoch": best_epoch, "best_validation_macro_f1": best_score,
                "baseline_validation_macro_f1": baseline["selection_macro_f1"], "test_evaluated": False})
    print(f"完成。最佳轮次：{best_epoch}（0代表初始模型仍最佳）。模型：{args.output_dir / 'best_model.pt'}")
    print("测试集尚未推理。训练结束后用相同output-dir加--evaluate-test评估两个模型。")


if __name__ == "__main__":
    main()
