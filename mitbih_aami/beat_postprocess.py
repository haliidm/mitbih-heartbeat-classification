"""Shared V5 label cleanup for the desktop viewer and evaluator."""

import numpy as np
from scipy.signal import find_peaks

from .label_cleanup import postprocess_pointwise_labels


SHORT_REGION_MS = 180.0
NORMAL_REGION_MAX_MS = 500.0
CLOSE_GAP_MS = 180.0
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0
RR_RELATIVE_TOLERANCE = 0.25
MIN_RR_INTERVALS = 3

RULE_DESCRIPTION = (
    "V5不填补背景缝隙；保留小于100毫秒过滤及V4上下文短段过滤。"
    "不超过500毫秒的区域按中点定位；更长区域在预处理ECG绝对幅值上寻找候选峰，"
    "峰间至少200毫秒，高度至少为区域绝对幅值95分位数的45%，"
    "突出度至少为该尺度的35%且不低于0.3标准化幅值。"
    "至少两个候选峰才按相邻峰中点分割，拆分区域用候选峰定位；"
    "候选峰不足两个时保留一个区域并按区域中点定位，不强行等分。"
    "每个区域N占比大于10%判N，否则按主类别；不使用ATR确定拆分点。"
)



def clean_labels_v5(labels, sample_rate):
    """Return cleaned labels and an audit of contextual short-region removal.

    Run once on the full record before cutting display windows. Decisions use
    the same unchanged list of regions, so removal order cannot affect them.
    """
    result = postprocess_pointwise_labels(labels, sample_rate=sample_rate, max_gap_ms=0)
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


def decode_v5(labels, signal, sample_rate):
    """Return cleanup labels, classified regions, predicted beats and audit."""
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.shape != np.asarray(labels).shape or not np.isfinite(signal).all():
        raise ValueError("ECG must be finite, one-dimensional and aligned with labels")
    cleaned, audit = clean_labels_v5(labels, sample_rate)
    edges = np.diff(np.r_[False, cleaned != 0, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    regions, beats, split_audit = [], [], []
    for start, end in zip(starts, ends):
        start, end = int(start), int(end)
        peaks = np.array([], dtype=np.int64)
        long_region = end - start > round(.5 * sample_rate)
        if long_region:
            left, right = max(0,start-1), min(len(signal),end+1)
            magnitude = np.abs(signal[left:right])
            scale = float(np.percentile(np.abs(signal[start:end]),95))
            peaks, _ = find_peaks(magnitude, distance=max(1,round(.2*sample_rate)),
                                  height=.45*scale, prominence=max(.3,.35*scale))
            peaks = peaks + left
            peaks = peaks[(peaks>=start)&(peaks<end)]
        if len(peaks) >= 2:
            boundaries = [start, *(((peaks[:-1]+peaks[1:])//2)+1).tolist(), end]
            parts = [(boundaries[i],boundaries[i+1],int(peak)) for i,peak in enumerate(peaks)]
        else:
            parts = [(start,end,(start+end-1)//2)]
        if long_region:
            split_audit.append({"start_seconds":start/sample_rate,"end_seconds":end/sample_rate,
                                "candidate_peaks_seconds":(peaks/sample_rate).tolist(),
                                "output_beats":len(parts),"split":len(peaks)>=2})
        for left,right,position in parts:
            counts = np.bincount(cleaned[left:right].astype(np.int64),minlength=6)
            counts[0]=0
            category = 1 if counts[1]/(right-left)>.1 else int(np.argmax(counts))
            regions.append((left,right,category))
            beats.append((position,category))
    audit['long_regions']=split_audit
    return cleaned, regions, beats, audit


def display_labels_v5(labels, signal, sample_rate):
    cleaned, regions, beats, audit = decode_v5(labels,signal,sample_rate)
    colored = cleaned.copy()
    for left,right,category in regions:
        colored[left:right]=category
    return colored, audit
