"""Evaluate full-record WFDB heartbeat detections from pointwise predictions."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .label_config import FS, LABEL_RADIUS_MS, NUM_CLASSES, WINDOW_SAMPLES
from .ecg_data import available_record_names, load_record_for_inference
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer
from .legacy_beat_postprocess_v4 import RULE_DESCRIPTION, refine_pointwise_labels_v4, classified_regions_v4


Beat = Tuple[int, int]
REPORT_CLASSES = (
    ("R", "R峰"),
    ("N", "N"),
    ("S", "S"),
    ("V", "V"),
    ("F", "F"),
    ("Q", "Q"),
)
COUNT_FIELDS = (
    "RTP", "RFP", "RFN",
    "NTP", "NFP", "NFN",
    "STP", "SFP", "SFN",
    "VTP", "VFP", "VFN",
    "QTP", "QFP", "QFN",
    "FTP", "FFP", "FFN",
)


def extract_predicted_beats(pointwise_labels: np.ndarray, already_processed=False) -> List[Beat]:
    """Clean pointwise labels, then split long runs and apply N dominance."""
    if already_processed:
        labels = np.asarray(pointwise_labels, dtype=np.int64)
    else:
        labels = refine_pointwise_labels_v4(pointwise_labels, sample_rate=FS)[0].astype(np.int64, copy=False)
    return [((start + end - 1) // 2, category)
            for start, end, category in classified_regions_v4(labels, sample_rate=FS)]



def match_by_position(
    reference: Sequence[Beat],
    predicted: Sequence[Beat],
    tolerance_samples: int,
):
    """Greedily match sorted events one-to-one when their centers are within tolerance."""
    refs = sorted(reference, key=lambda beat: beat[0])
    preds = sorted(predicted, key=lambda beat: beat[0])
    matches = []
    missed_reference = []
    extra_predictions = []
    i = j = 0

    while i < len(refs) and j < len(preds):
        difference = preds[j][0] - refs[i][0]
        if difference < -tolerance_samples:
            extra_predictions.append(preds[j])
            j += 1
        elif difference > tolerance_samples:
            missed_reference.append(refs[i])
            i += 1
        else:
            matches.append((refs[i], preds[j]))
            i += 1
            j += 1

    missed_reference.extend(refs[i:])
    extra_predictions.extend(preds[j:])
    return matches, missed_reference, extra_predictions


def _empty_counts() -> Dict[str, Dict[str, int]]:
    return {
        name: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for _, name in REPORT_CLASSES
    }


def count_record_events(reference: Sequence[Beat], predicted: Sequence[Beat], tolerance_samples: int):
    matches, missed_reference, extra_predictions = match_by_position(
        reference, predicted, tolerance_samples
    )
    counts = _empty_counts()
    counts["R峰"]["TP"] = len(matches)
    counts["R峰"]["FN"] = len(missed_reference)
    counts["R峰"]["FP"] = len(extra_predictions)

    # For N/S/V/F/Q, a location-matched but mistyped beat is an FN for its
    # reference class and an FP for its predicted class.
    for (_, reference_class), (_, predicted_class) in matches:
        for class_id, (_, class_name) in enumerate(REPORT_CLASSES[1:], start=1):
            is_reference_class = reference_class == class_id
            is_predicted_class = predicted_class == class_id
            if is_reference_class and is_predicted_class:
                counts[class_name]["TP"] += 1
            elif is_reference_class:
                counts[class_name]["FN"] += 1
            elif is_predicted_class:
                counts[class_name]["FP"] += 1
            else:
                counts[class_name]["TN"] += 1

    for _, reference_class in missed_reference:
        for class_id, (_, class_name) in enumerate(REPORT_CLASSES[1:], start=1):
            if reference_class == class_id:
                counts[class_name]["FN"] += 1
            else:
                counts[class_name]["TN"] += 1

    for _, predicted_class in extra_predictions:
        for class_id, (_, class_name) in enumerate(REPORT_CLASSES[1:], start=1):
            if predicted_class == class_id:
                counts[class_name]["FP"] += 1
            else:
                counts[class_name]["TN"] += 1

    return counts, len(reference), len(predicted), len(matches)


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def calculate_metrics(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for _, class_name in REPORT_CLASSES:
        item = counts[class_name]
        tp, fp, fn, tn = item["TP"], item["FP"], item["FN"], item["TN"]
        sensitivity = _ratio(tp, tp + fn)
        # Every recognized ATR beat is QRS-positive; there are no true-negative
        # beat candidates for the aggregate R-peak event detector.
        specificity = None if class_name == "R峰" else _ratio(tn, tn + fp)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        metrics[class_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "敏感度": sensitivity,
            "特异性": specificity,
            "F1分数": f1,
        }
    return metrics


def _count_row(record_name: str, counts: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    row: Dict[str, object] = {"记录": record_name}
    for prefix, class_name in REPORT_CLASSES:
        row[f"{prefix}TP"] = counts[class_name]["TP"]
        row[f"{prefix}FP"] = counts[class_name]["FP"]
        row[f"{prefix}FN"] = counts[class_name]["FN"]
    return row


def _write_counts_csv(path: Path, rows):
    fields = ("记录",) + COUNT_FIELDS
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_rate(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def _write_metrics_csv(path: Path, rows):
    fields = ("记录", "类别", "TP", "FP", "FN", "TN", "敏感度", "特异性", "F1分数")
    rate_fields = ("敏感度", "特异性", "F1分数")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in rate_fields:
                formatted[field] = _format_rate(formatted[field])
            writer.writerow(formatted)


def _markdown_table(headers, rows, left_columns=1):
    lines = [
        "| " + " | ".join(str(value) for value in headers) + " |",
        "| " + " | ".join(
            ":---" if index < left_columns else "---:"
            for index in range(len(headers))
        ) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return lines


def _write_markdown_report(path: Path, count_rows, metric_rows, tolerance_ms):
    count_headers = ("记录",) + COUNT_FIELDS
    count_data = [
        [row[field] for field in count_headers]
        for row in count_rows
    ]
    metric_headers = ("记录", "类别", "TP", "FP", "FN", "TN", "敏感度", "特异性", "F1分数")
    metric_data = [
        [
            row["记录"], row["类别"], row["TP"], row["FP"], row["FN"], row["TN"],
            _format_rate(row["敏感度"]),
            _format_rate(row["特异性"]),
            _format_rate(row["F1分数"]),
        ]
        for row in metric_rows
    ]
    lines = [
        "# 完整记录心搏评估报告（规则四）",
        "",
        f"位置匹配阈值：±{tolerance_ms:.0f} ms",
        RULE_DESCRIPTION,
        "混合区域判类：区域内 N 类点占比大于 10% 时，该区域按 N 类计。",
        "长区域拆分：连续非零区域时长大于 500 ms 时，在区域中点切分为两个候选心搏区域。",
        "",
        "## 心搏事件计数",
        "",
        *_markdown_table(count_headers, count_data),
        "",
        "## 分类指标",
        "",
        *_markdown_table(metric_headers, metric_data, left_columns=2),
        "",
        "注：R 峰事件级特异性为“—”，因为 ATR 心搏标注均为 QRS 阳性事件，"
        "没有事件级真实阴性样本。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_aligned_table(headers, rows):
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(" | ".join(value.rjust(widths[index]) if index else value.ljust(widths[index])
                        for index, value in enumerate(row)))


def _write_record_info_csv(path: Path, rows):
    fields = ("记录", "源采样率（Hz）", "模型采样率（Hz）", "时长（秒）", "参考心搏数")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append_database_info(markdown_path: Path, database_dir: Path, rows):
    headers = ("记录", "源采样率（Hz）", "模型采样率（Hz）", "时长（秒）", "参考心搏数")
    data = [
        [row[field] for field in headers]
        for row in rows
    ]
    lines = [
        "",
        "## 数据库与采样率",
        "",
        f"数据库目录：{database_dir.resolve()}",
        f"模型输入采样率：{FS} Hz",
        "参考心搏数仅包含当前符号映射支持的 ATR 心搏标注。",
        "",
        *_markdown_table(headers, data),
        "",
    ]
    with markdown_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))

def parse_args():
    parser = argparse.ArgumentParser(
        description="对 WFDB 完整记录进行逐点推理，并按连续非零区域统计心搏事件。"
    )
    parser.add_argument("--database-dir", type=Path, default=Path("mit-bih-arrhythmia-database-1.0.0"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练输出的 best_model.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/full_record_evaluation_v4"))
    parser.add_argument(
        "--records", nargs="*", default=None,
        help="可选：只评估指定记录，例如 --records 100 101 或 --records e0103；默认评估目录中全部记录。",
    )
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument("--batch-windows", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms 必须大于或等于 0。")
    if args.batch_windows < 1:
        raise ValueError("--batch-windows 必须大于 0。")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到模型文件：{args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{'GPU' if device.type == 'cuda' else 'CPU'}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_radius = checkpoint.get("label_radius_ms")
    if checkpoint_radius is not None and int(checkpoint_radius) != LABEL_RADIUS_MS:
        raise ValueError(
            f"模型标签半径为 {checkpoint_radius} ms，当前项目要求 ±{LABEL_RADIUS_MS} ms。"
        )

    input_samples = int(checkpoint.get("input_samples", WINDOW_SAMPLES))
    model = MultiScaleCNNTransformer(input_samples=input_samples).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_names = available_record_names(args.database_dir)
    if args.records is None:
        record_names = all_names
    else:
        requested = [str(name) for name in args.records]
        missing = [name for name in requested if name not in all_names]
        if missing:
            raise FileNotFoundError(f"数据库中找不到记录：{', '.join(missing)}")
        record_names = requested
    if not record_names:
        raise ValueError("没有可评估的记录。")

    tolerance_samples = round(args.tolerance_ms * FS / 1000.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count_rows = []
    metric_rows = []
    record_info_rows = []
    removal_rows = []
    cleanup_rows = []
    total_counts = _empty_counts()
    total_reference = total_predicted = total_matched = 0

    for record_name in record_names:
        print(f"正在推理记录 {record_name}……")
        record, source_fs = load_record_for_inference(args.database_dir, record_name)
        record_info_rows.append({
            "记录": record_name,
            "源采样率（Hz）": source_fs,
            "模型采样率（Hz）": FS,
            "时长（秒）": round(len(record.signal) / FS, 3),
            "参考心搏数": len(record.events),
        })
        logits = predict_record(
            model, record, device, batch_windows=args.batch_windows
        )
        pointwise_prediction = np.argmax(logits, axis=0).astype(np.uint8)
        corrected_labels, cleanup = refine_pointwise_labels_v4(pointwise_prediction, sample_rate=FS)
        predicted_beats = extract_predicted_beats(corrected_labels, already_processed=True)
        removal_rows.extend({"记录": record_name, **row} for row in cleanup["removed_regions"])
        cleanup_rows.append({"记录": record_name,
                             "典型RR（毫秒）": cleanup["typical_rr_ms"],
                             "额外删除短区域数": len(cleanup["removed_regions"])})
        print(f"  第四版额外删除短区域 {len(cleanup['removed_regions'])} 个")
        reference_beats = list(record.events)

        counts, ref_count, pred_count, matched_count = count_record_events(
            reference_beats, predicted_beats, tolerance_samples
        )
        count_rows.append(_count_row(record_name, counts))
        total_reference += ref_count
        total_predicted += pred_count
        total_matched += matched_count

        for _, class_name in REPORT_CLASSES:
            for key in ("TP", "FP", "FN", "TN"):
                total_counts[class_name][key] += counts[class_name][key]
            metrics = calculate_metrics(counts)[class_name]
            metric_rows.append({
                "记录": record_name,
                "类别": class_name,
                **metrics,
            })

        print(
            f"  参考心搏 {ref_count}，预测区域 {pred_count}，"
            f"位置匹配 {matched_count}"
        )

    count_rows.append(_count_row("Total", total_counts))
    total_metrics = calculate_metrics(total_counts)
    for _, class_name in REPORT_CLASSES:
        metric_rows.append({
            "记录": "Total",
            "类别": class_name,
            **total_metrics[class_name],
        })

    counts_path = args.output_dir / "心搏事件计数.csv"
    metrics_path = args.output_dir / "分类指标.csv"
    _write_counts_csv(counts_path, count_rows)
    _write_metrics_csv(metrics_path, metric_rows)
    markdown_path = args.output_dir / "事件评估报告.md"
    _write_markdown_report(markdown_path, count_rows, metric_rows, args.tolerance_ms)
    info_path = args.output_dir / "记录信息.csv"
    _write_record_info_csv(info_path, record_info_rows)
    _append_database_info(markdown_path, args.database_dir, record_info_rows)
    removal_path = args.output_dir / "第四版删除短区域明细.csv"
    removal_fields = ("记录", "起始时间（秒）", "结束时间（秒，右端不含）", "区域时长（毫秒）",
                      "左侧间隔（毫秒）", "右侧间隔（毫秒）", "两侧中心间距（毫秒）", "典型RR（毫秒）")
    with removal_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=removal_fields)
        writer.writeheader()
        writer.writerows(removal_rows)
    with markdown_path.open("a", encoding="utf-8") as file:
        file.write("\n## 第四版额外删除统计\n\n")
        headers = ("记录", "典型RR（毫秒）", "额外删除短区域数")
        rows = [[row["记录"], "不足以估计" if row["典型RR（毫秒）"] is None
                 else f"{row['典型RR（毫秒）']:.2f}", row["额外删除短区域数"]] for row in cleanup_rows]
        file.write("\n".join(_markdown_table(headers, rows)) + "\n")
        file.write("\n模型文件：" + str(args.checkpoint.resolve()) + "\n")
        file.write("\n删除明细：第四版删除短区域明细.csv。这里仅统计第四版额外删除，不包含基础的小于100毫秒过滤。\n")

    print("\n心搏事件计数（位置匹配容差 ±{:.0f} ms；规则四）".format(args.tolerance_ms))
    _print_aligned_table(
        ("记录",) + COUNT_FIELDS,
        [[row[field] for field in ("记录",) + COUNT_FIELDS] for row in count_rows],
    )

    print("\n总体指标")
    metric_headers = ("类别", "TP", "FP", "FN", "TN", "敏感度", "特异性", "F1分数")
    total_metric_rows = []
    for _, class_name in REPORT_CLASSES:
        item = total_metrics[class_name]
        total_metric_rows.append([
            class_name, item["TP"], item["FP"], item["FN"], item["TN"],
            _format_rate(item["敏感度"]),
            _format_rate(item["特异性"]),
            _format_rate(item["F1分数"]),
        ])
    _print_aligned_table(metric_headers, total_metric_rows)

    print(
        f"\n全体统计：参考心搏 {total_reference}，预测区域 {total_predicted}，"
        f"位置匹配 {total_matched}。"
    )
    print(f"记录信息：{info_path.resolve()}")
    print(f"计数表：{counts_path.resolve()}")
    print(f"指标表：{metrics_path.resolve()}")
    print(f"对齐报告：{markdown_path.resolve()}")
    print(RULE_DESCRIPTION)
    print(f"删除明细：{removal_path.resolve()}")
    print("混合区域N占比大于10%时归为N；长于500毫秒的区域仍从中点拆为两段。")
    print("说明：R峰事件级特异性记为“—”，因为 ATR 中所有已标注心搏都属于 QRS，"
          "没有可定义的真实阴性心搏事件。")

if __name__ == "__main__":
    main()
