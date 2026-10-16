"""Evaluate full-record WFDB heartbeat detections from pointwise predictions."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .label_config import FS, LABEL_RADIUS_MS, NUM_CLASSES, WINDOW_SAMPLES
from .ecg_data import available_record_names, load_record_for_inference
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer
from .beat_postprocess import RULE_DESCRIPTION, decode_v5


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
    """Count QRS detection and build a 5x5 class matrix for position matches."""
    matches, missed_reference, extra_predictions = match_by_position(
        reference, predicted, tolerance_samples
    )
    counts = _empty_counts()
    counts["R峰"]["TP"] = len(matches)
    counts["R峰"]["FN"] = len(missed_reference)
    counts["R峰"]["FP"] = len(extra_predictions)
    confusion = np.zeros((5, 5), dtype=np.int64)

    for (_, reference_class), (_, predicted_class) in matches:
        if 1 <= reference_class <= 5 and 1 <= predicted_class <= 5:
            confusion[reference_class - 1, predicted_class - 1] += 1

    # Keep the event-count table: correct type matches are TP, type confusions
    # are class FP/FN, and location misses/extras count against their class.
    for (_, reference_class), (_, predicted_class) in matches:
        for class_id, (_, class_name) in enumerate(REPORT_CLASSES[1:], start=1):
            if reference_class == class_id and predicted_class == class_id:
                counts[class_name]["TP"] += 1
            elif reference_class == class_id:
                counts[class_name]["FN"] += 1
            elif predicted_class == class_id:
                counts[class_name]["FP"] += 1
    for _, reference_class in missed_reference:
        if 1 <= reference_class <= 5:
            counts[REPORT_CLASSES[reference_class][1]]["FN"] += 1
    for _, predicted_class in extra_predictions:
        if 1 <= predicted_class <= 5:
            counts[REPORT_CLASSES[predicted_class][1]]["FP"] += 1

    return counts, confusion, len(reference), len(predicted), len(matches)


def detection_metrics(counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    item = counts["R峰"]
    tp, fp, fn = item["TP"], item["FP"], item["FN"]
    precision = _ratio(tp, tp + fp)
    sensitivity = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": "", "阳性预测值": precision,
            "敏感度": sensitivity, "特异性": None, "F1分数": f1}


def classification_metrics(confusion: np.ndarray) -> Dict[str, Dict[str, float]]:
    total = int(confusion.sum())
    metrics = {}
    for index, (_, class_name) in enumerate(REPORT_CLASSES[1:]):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum()) - tp
        fp = int(confusion[:, index].sum()) - tp
        tn = total - tp - fn - fp
        precision = _ratio(tp, tp + fp)
        sensitivity = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        metrics[class_name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                               "阳性预测值": precision, "敏感度": sensitivity,
                               "特异性": specificity, "F1分数": f1}
    return metrics

def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None



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
    fields = ("记录", "类别", "TP", "FP", "FN", "TN", "阳性预测值", "敏感度", "特异性", "F1分数")
    rate_fields = ("阳性预测值", "敏感度", "特异性", "F1分数")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {field: row.get(field, "") for field in fields}
            for field in rate_fields:
                formatted[field] = _format_rate(formatted[field])
            writer.writerow(formatted)


def _write_confusion_csv(path: Path, matrices):
    fields = ("记录", "真实类别", "预测N", "预测S", "预测V", "预测F", "预测Q")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record_name, matrix in matrices.items():
            for row_index, (_, class_name) in enumerate(REPORT_CLASSES[1:]):
                writer.writerow({"记录": record_name, "真实类别": class_name,
                                 **{fields[col+2]: int(matrix[row_index, col]) for col in range(5)}})

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


def _write_markdown_report(path: Path, count_rows, metric_rows, confusion_matrices, tolerance_ms):
    count_headers = ("记录",) + COUNT_FIELDS
    count_data = [[row[field] for field in count_headers] for row in count_rows]
    metric_headers = ("记录", "类别", "TP", "FP", "FN", "TN", "阳性预测值", "敏感度", "特异性", "F1分数")
    metric_data = []
    for row in metric_rows:
        metric_data.append([row.get("记录", ""), row.get("类别", ""), row.get("TP", ""),
            row.get("FP", ""), row.get("FN", ""), row.get("TN", ""),
            *[_format_rate(row.get(key)) for key in ("阳性预测值", "敏感度", "特异性", "F1分数")]])
    matrix = confusion_matrices["Total"]
    matrix_headers = ("参考类别/预测类别", "N", "S", "V", "F", "Q")
    matrix_data = [[REPORT_CLASSES[i+1][1], *[int(value) for value in matrix[i]]] for i in range(5)]
    lines = [
        "# 完整记录心搏评估报告（规则五）", "", f"位置匹配阈值：±{tolerance_ms:.0f} ms",
        RULE_DESCRIPTION,
        "R 峰检测只统计 TP、FP、FN，并计算阳性预测值、敏感度和 F1；没有定义真实阴性候选，因此不计算特异性。",
        "五分类混淆矩阵仅包含位置匹配的心搏；矩阵行为参考类别，列为预测类别。分类指标按该矩阵计算。位置未匹配的漏检/多检在 R 峰检测及事件计数中单独体现。",
        "混合区域判类：区域内 N 类点占比大于 10% 时，该区域按 N 类计。",
        "长区域按ECG候选峰拆分，候选峰不足两个不强行拆分；详见各记录long_regions.json。", "",
        "## 心搏事件计数", "", *_markdown_table(count_headers, count_data), "",
        "## 五分类混淆矩阵（位置匹配心搏）", "", *_markdown_table(matrix_headers, matrix_data), "",
        "## R 峰检测与分类指标", "", *_markdown_table(metric_headers, metric_data, left_columns=2), "",
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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/full_record_evaluation_v5"))
    parser.add_argument(
        "--records", nargs="*", default=None,
        help="可选：只评估指定记录，例如 --records 100 101 或 --records e0103；默认评估目录中全部记录。",
    )
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument("--batch-windows", type=int, default=None, help="默认根据窗口长度自动设置：16秒16批、32秒2批、64秒1批")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms 必须大于或等于 0。")
    if args.batch_windows is not None and args.batch_windows < 1:
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
    if args.batch_windows is None:
        args.batch_windows = 16 if input_samples <= WINDOW_SAMPLES else 2 if input_samples <= 32 * FS else 1
    print(f"模型输入窗口：{input_samples / FS:g} 秒；整条记录推理批量：{args.batch_windows} 个窗口")
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
    confusion_matrices = {}
    record_info_rows = []
    removal_rows = []
    cleanup_rows = []
    total_counts = _empty_counts()
    total_confusion = np.zeros((5, 5), dtype=np.int64)
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
        corrected_labels, regions, predicted_beats, cleanup = decode_v5(pointwise_prediction, record.signal, sample_rate=FS)
        (args.output_dir / f"{record_name}_long_regions.json").write_text(json.dumps(cleanup["long_regions"], indent=2, ensure_ascii=False), encoding="utf-8")
        removal_rows.extend({"记录": record_name, **row} for row in cleanup["removed_regions"])
        cleanup_rows.append({"记录": record_name,
                             "典型RR（毫秒）": cleanup["typical_rr_ms"],
                             "额外删除短区域数": len(cleanup["removed_regions"])})
        print(f"  第五版额外删除短区域 {len(cleanup['removed_regions'])} 个")
        reference_beats = list(record.events)

        counts, confusion, ref_count, pred_count, matched_count = count_record_events(
            reference_beats, predicted_beats, tolerance_samples
        )
        confusion_matrices[record_name] = confusion
        total_confusion += confusion
        count_rows.append(_count_row(record_name, counts))
        total_reference += ref_count
        total_predicted += pred_count
        total_matched += matched_count

        for _, class_name in REPORT_CLASSES:
            for key in ("TP", "FP", "FN", "TN"):
                total_counts[class_name][key] += counts[class_name][key]
        metric_rows.append({"记录": record_name, "类别": "R峰", **detection_metrics(counts)})
        for class_name, metrics in classification_metrics(confusion).items():
            metric_rows.append({"记录": record_name, "类别": class_name, **metrics})

        print(
            f"  参考心搏 {ref_count}，预测区域 {pred_count}，"
            f"位置匹配 {matched_count}"
        )

    count_rows.append(_count_row("Total", total_counts))
    confusion_matrices["Total"] = total_confusion
    metric_rows.append({"记录": "Total", "类别": "R峰", **detection_metrics(total_counts)})
    total_metrics = classification_metrics(total_confusion)
    for class_name, metrics in total_metrics.items():
        metric_rows.append({"记录": "Total", "类别": class_name, **metrics})

    counts_path = args.output_dir / "心搏事件计数.csv"
    metrics_path = args.output_dir / "分类指标.csv"
    _write_counts_csv(counts_path, count_rows)
    _write_metrics_csv(metrics_path, metric_rows)
    confusion_path = args.output_dir / "五分类混淆矩阵.csv"
    _write_confusion_csv(confusion_path, confusion_matrices)
    markdown_path = args.output_dir / "事件评估报告.md"
    _write_markdown_report(markdown_path, count_rows, metric_rows, confusion_matrices, args.tolerance_ms)
    info_path = args.output_dir / "记录信息.csv"
    _write_record_info_csv(info_path, record_info_rows)
    _append_database_info(markdown_path, args.database_dir, record_info_rows)
    removal_path = args.output_dir / "第五版删除短区域明细.csv"
    removal_fields = ("记录", "起始时间（秒）", "结束时间（秒，右端不含）", "区域时长（毫秒）",
                      "左侧间隔（毫秒）", "右侧间隔（毫秒）", "两侧中心间距（毫秒）", "典型RR（毫秒）")
    with removal_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=removal_fields)
        writer.writeheader()
        writer.writerows(removal_rows)
    with markdown_path.open("a", encoding="utf-8") as file:
        file.write("\n## 第五版额外删除统计\n\n")
        headers = ("记录", "典型RR（毫秒）", "额外删除短区域数")
        rows = [[row["记录"], "不足以估计" if row["典型RR（毫秒）"] is None
                 else f"{row['典型RR（毫秒）']:.2f}", row["额外删除短区域数"]] for row in cleanup_rows]
        file.write("\n".join(_markdown_table(headers, rows)) + "\n")
        file.write("\n模型文件：" + str(args.checkpoint.resolve()) + "\n")
        file.write("\n删除明细：第五版删除短区域明细.csv。这里仅统计第五版额外删除，不包含基础的小于100毫秒过滤。\n")

    print("\n心搏事件计数（位置匹配容差 ±{:.0f} ms；规则五）".format(args.tolerance_ms))
    _print_aligned_table(
        ("记录",) + COUNT_FIELDS,
        [[row[field] for field in ("记录",) + COUNT_FIELDS] for row in count_rows],
    )

    print("\n总体 R 峰检测指标")
    r_metrics = detection_metrics(total_counts)
    _print_aligned_table(
        ("TP", "FP", "FN", "阳性预测值", "敏感度", "F1分数"),
        [[r_metrics["TP"], r_metrics["FP"], r_metrics["FN"],
          _format_rate(r_metrics["阳性预测值"]), _format_rate(r_metrics["敏感度"]),
          _format_rate(r_metrics["F1分数"])]]
    )
    print("\n总体五分类混淆矩阵（行=参考类别，列=预测类别；仅统计位置匹配心搏）")
    matrix_headers = ("参考 \\ 预测", "N", "S", "V", "F", "Q")
    matrix_rows = [[REPORT_CLASSES[i+1][1], *[int(value) for value in total_confusion[i]]]
                   for i in range(5)]
    _print_aligned_table(matrix_headers, matrix_rows)
    print("\n分类指标（根据五分类混淆矩阵计算）")
    metric_headers = ("类别", "TP", "FP", "FN", "TN", "阳性预测值", "敏感度", "特异性", "F1分数")
    class_metric_rows = []
    for class_name, item in total_metrics.items():
        class_metric_rows.append([class_name, item["TP"], item["FP"], item["FN"], item["TN"],
            _format_rate(item["阳性预测值"]), _format_rate(item["敏感度"]),
            _format_rate(item["特异性"]), _format_rate(item["F1分数"])])
    _print_aligned_table(metric_headers, class_metric_rows)

    print(
        f"\n全体统计：参考心搏 {total_reference}，预测区域 {total_predicted}，"
        f"位置匹配 {total_matched}。"
    )
    print(f"记录信息：{info_path.resolve()}")
    print(f"计数表：{counts_path.resolve()}")
    print(f"指标表：{metrics_path.resolve()}")
    print(f"五分类混淆矩阵：{confusion_path.resolve()}")
    print(f"对齐报告：{markdown_path.resolve()}")
    print(RULE_DESCRIPTION)
    print(f"删除明细：{removal_path.resolve()}")
    print("混合区域N占比大于10%时归为N；长区域采用ECG候选峰拆分。")
    print("R 峰检测的阳性预测值、敏感度和 F1 仅由 TP、FP、FN 计算；分类指标仅统计位置匹配的心搏。")

if __name__ == "__main__":
    main()
