"""Shared V4 label cleanup for the desktop viewer and evaluator."""

import numpy as np

from .label_cleanup import postprocess_pointwise_labels


SHORT_REGION_MS = 180.0
NORMAL_REGION_MAX_MS = 500.0
CLOSE_GAP_MS = 180.0
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
RR_RELATIVE_TOLERANCE = 0.25
MIN_RR_INTERVALS = 3

RULE_DESCRIPTION = (
    "先填补同主类别区域之间不超过50毫秒的背景缝隙，再删除所有小于100毫秒的非零区域。"
    "V4额外检查夹在两个180至500毫秒区域之间、短于180毫秒的非零区域；"
    "两侧区域中心间距须在300至2000毫秒内，且处于典型RR的75%至125%范围，"
    "短段与任一侧区域的边缘间隔须小于180毫秒，满足时将短段置为背景。"
    "典型RR为整条记录中相邻180至500毫秒区域中心间距的中位数，"
    "仅纳入300至2000毫秒间距；不足3个间距时不执行额外删除。"
    "这里的区域宽度条件与N类别无关；规则只使用预测标签，不使用参考标注。"
)


def refine_pointwise_labels_v4(labels, sample_rate):
    """Return cleaned labels and an audit of contextual short-region removal.

    Run once on the full record before cutting display windows. Decisions use
    the same unchanged list of regions, so removal order cannot affect them.
    """
    result = postprocess_pointwise_labels(labels, sample_rate=sample_rate)
    details = {"typical_rr_ms": None, "removed_regions": []}
    active = result != 0
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if len(starts) < 3:
        return result, details
    widths_ms = (ends - starts) * 1000.0 / sample_rate
    centers = (starts + ends - 1) / 2.0
    normal = (widths_ms >= SHORT_REGION_MS) & (widths_ms <= NORMAL_REGION_MAX_MS)
    intervals = np.diff(centers[normal]) * 1000.0 / sample_rate
    intervals = intervals[(intervals >= RR_MIN_MS) & (intervals <= RR_MAX_MS)]
    if len(intervals) < MIN_RR_INTERVALS:
        return result, details
    typical_rr = float(np.median(intervals))
    details["typical_rr_ms"] = typical_rr

    for index in range(1, len(starts) - 1):
        if widths_ms[index] >= SHORT_REGION_MS:
            continue
        if not (normal[index - 1] and normal[index + 1]):
            continue
        outer_rr = (centers[index + 1] - centers[index - 1]) * 1000.0 / sample_rate
        if not (RR_MIN_MS <= outer_rr <= RR_MAX_MS):
            continue
        if not ((1 - RR_RELATIVE_TOLERANCE) * typical_rr <= outer_rr
                <= (1 + RR_RELATIVE_TOLERANCE) * typical_rr):
            continue
        left_gap = (starts[index] - ends[index - 1]) * 1000.0 / sample_rate
        right_gap = (starts[index + 1] - ends[index]) * 1000.0 / sample_rate
        if min(left_gap, right_gap) >= CLOSE_GAP_MS:
            continue
        details["removed_regions"].append({
            "起始时间（秒）": float(starts[index] / sample_rate),
            "结束时间（秒，右端不含）": float(ends[index] / sample_rate),
            "区域时长（毫秒）": float(widths_ms[index]),
            "左侧间隔（毫秒）": float(left_gap),
            "右侧间隔（毫秒）": float(right_gap),
            "两侧中心间距（毫秒）": float(outer_rr),
            "典型RR（毫秒）": typical_rr,
        })
        result[starts[index]:ends[index]] = 0
    return result, details


def postprocess_pointwise_labels_v4(labels, sample_rate):
    return refine_pointwise_labels_v4(labels, sample_rate)[0]


def classified_regions_v4(cleaned_labels, sample_rate):
    """Return (start, exclusive end, class) using the V4 evaluation rules.

    Input must already have undergone full-record V4 cleanup. Long runs are
    split once at the midpoint, exactly as in the evaluator.
    """
    labels = np.asarray(cleaned_labels, dtype=np.int64)
    if labels.ndim != 1 or sample_rate <= 0:
        raise ValueError("Expected one-dimensional labels and positive sample_rate")
    edges = np.diff(np.r_[False, labels != 0, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    regions = []
    for start, end in zip(starts, ends):
        start, end = int(start), int(end)
        if end - start > round(500 * sample_rate / 1000):
            middle = start + (end - start) // 2
            parts = ((start, middle), (middle, end))
        else:
            parts = ((start, end),)
        for left, right in parts:
            counts = np.bincount(labels[left:right], minlength=6)
            counts[0] = 0
            category = 1 if counts[1] / (right - left) > 0.10 else int(np.argmax(counts))
            regions.append((left, right, category))
    return regions


def color_final_regions_v4(cleaned_labels, sample_rate):
    """Paint each evaluated heartbeat region with its final assigned class."""
    result = np.asarray(cleaned_labels).copy()
    for start, end, category in classified_regions_v4(cleaned_labels, sample_rate):
        result[start:end] = category
    return result
