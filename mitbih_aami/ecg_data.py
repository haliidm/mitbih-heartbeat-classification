"""MIT-BIH loading, preprocessing, pointwise labels, and event-centered windows."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from torch.utils.data import Dataset
import wfdb

from .label_config import (
    CENTER_CLASSES,
    FS,
    MAX_LABEL_RADIUS,
    PEAK_REFINEMENT_RADIUS,
    PEAK_REFINEMENT_RADIUS_MS,
    PAD_SAMPLES,
    SYMBOL_TO_CLASS,
    WINDOW_HALF,
    WINDOW_SAMPLES,
)


@dataclass
class RecordData:
    name: str
    lead_name: str
    signal: np.ndarray
    signal_padded: np.ndarray
    target: np.ndarray
    target_padded: np.ndarray
    events: List[Tuple[int, int]]
    original_events: List[Tuple[int, int]]
    refinement_info: List[dict]


def available_record_names(data_dir: Path) -> List[str]:
    """Return every record with a WFDB header in the database directory."""
    names = sorted(path.stem for path in Path(data_dir).glob("*.hea"))
    if not names:
        raise FileNotFoundError(f"No .hea files found in {data_dir}")
    return names


def _filter_and_normalize(signal: np.ndarray, fs: int) -> np.ndarray:
    """Apply 0.67-40 Hz filtering and full-record z-score normalization."""
    if fs != FS:
        raise ValueError(f"Expected {FS} Hz MIT-BIH signal, got {fs} Hz.")
    sos = butter(4, (0.67, 40.0), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal.astype(np.float64, copy=False))
    mean = float(np.mean(filtered))
    std = float(np.std(filtered))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return ((filtered - mean) / std).astype(np.float32)


def _radius_for_event(index: int, events: Sequence[Tuple[int, int]]) -> int:
    sample = events[index][0]
    gaps = []
    if index > 0:
        gaps.append(sample - events[index - 1][0])
    if index + 1 < len(events):
        gaps.append(events[index + 1][0] - sample)
    if not gaps:
        return MAX_LABEL_RADIUS
    return min(MAX_LABEL_RADIUS, max(0, (min(gaps) - 1) // 2))


def _make_targets(length: int, events: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Create pointwise class regions centered on annotations without overlap."""
    target = np.zeros(length, dtype=np.uint8)
    for index, (sample, class_id) in enumerate(events):
        radius = _radius_for_event(index, events)
        left = max(0, sample - radius)
        right = min(length, sample + radius + 1)
        target[left:right] = class_id
    return target


def _select_lead(record) -> Tuple[int, str]:
    names = [str(name).strip().upper() for name in record.sig_name]
    for candidate in ("MLII", "V5", "V2"):
        if candidate in names:
            return names.index(candidate), candidate
    if not names:
        raise ValueError("Record has no named ECG leads.")
    return 0, names[0]


def _refine_events(
    signal: np.ndarray,
    events: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[dict]]:
    """Move annotations to the strongest nearby positive or negative peak.

    Candidate peak locations are restricted to 50 ms around each ATR sample
    and to the midpoint between adjacent annotations. One extra context sample
    on either side lets peak finding validate extrema at the search boundary.
    Amplitude selects the peak; second-difference sharpness breaks near-equal
    ties. If no local extremum exists, retain the ATR sample.
    """
    original_events = sorted(
        ((int(sample), int(class_id)) for sample, class_id in events),
        key=lambda item: item[0],
    )
    refined = []
    audit = []
    for index, (sample, class_id) in enumerate(original_events):
        left = max(0, sample - PEAK_REFINEMENT_RADIUS)
        right = min(len(signal), sample + PEAK_REFINEMENT_RADIUS + 1)
        if index > 0:
            left = max(left, (original_events[index - 1][0] + sample) // 2 + 1)
        if index + 1 < len(original_events):
            right = min(right, (sample + original_events[index + 1][0]) // 2 + 1)

        context_left = max(0, left - 1)
        context_right = min(len(signal), right + 1)
        context = signal[context_left:context_right]
        positive = context_left + find_peaks(context)[0]
        negative = context_left + find_peaks(-context)[0]
        candidates = np.unique(np.concatenate((positive, negative)))
        candidates = candidates[(candidates >= left) & (candidates < right)]

        def peak_sharpness(absolute_index):
            absolute_index = int(absolute_index)
            if absolute_index <= 0 or absolute_index >= len(signal) - 1:
                return 0.0
            return abs(
                2.0 * float(signal[absolute_index])
                - float(signal[absolute_index - 1])
                - float(signal[absolute_index + 1])
            )

        if not len(candidates):
            corrected = sample
            sharpness = peak_sharpness(sample)
        else:
            amplitudes = np.abs(signal[candidates])
            maximum = float(amplitudes.max())
            tied = candidates[amplitudes >= maximum * 0.99]
            corrected = max(
                (int(candidate) for candidate in tied),
                key=lambda candidate: (
                    peak_sharpness(candidate),
                    abs(float(signal[candidate])),
                    -abs(candidate - sample),
                ),
            )
            sharpness = peak_sharpness(corrected)

        amplitude = float(signal[corrected])
        refined.append((int(corrected), class_id))
        audit.append({
            "original_sample": int(sample),
            "corrected_sample": int(corrected),
            "class_id": class_id,
            "offset_samples": int(corrected - sample),
            "amplitude_z": amplitude,
            "polarity": "positive" if amplitude > 0 else "negative" if amplitude < 0 else "zero",
            "sharpness": sharpness,
            "search_left": int(left),
            "search_right_exclusive": int(right),
            "search_radius_ms": PEAK_REFINEMENT_RADIUS_MS,
        })
    return refined, audit


def _build_record(
    record_name: str,
    lead_name: str,
    raw_signal: np.ndarray,
    fs: int,
    events: List[Tuple[int, int]],
) -> RecordData:
    signal = _filter_and_normalize(raw_signal, fs)
    # Correct annotations and generate the full-record target before any windows are cut.
    original_events = sorted(events, key=lambda item: item[0])
    events, refinement_info = _refine_events(signal, original_events)
    target = _make_targets(len(signal), events)

    # Padding supports a full 16 s window centered on any sample, including
    # annotated beats close to either record boundary.
    signal_padded = np.pad(
        signal, (PAD_SAMPLES, PAD_SAMPLES), mode="reflect"
    ).astype(np.float32, copy=False)
    target_padded = np.pad(
        target, (PAD_SAMPLES, PAD_SAMPLES), mode="constant", constant_values=0
    ).astype(np.uint8, copy=False)
    return RecordData(
        name=record_name,
        lead_name=lead_name,
        signal=signal,
        signal_padded=signal_padded,
        target=target,
        target_padded=target_padded,
        events=events,
        original_events=original_events,
        refinement_info=refinement_info,
    )


def load_record(data_dir: Path, record_name: str) -> RecordData:
    """Load a record, preferring MLII and falling back to V5 or V2."""
    record_base = Path(data_dir) / record_name
    record = wfdb.rdrecord(str(record_base))
    lead_index, lead_name = _select_lead(record)
    if lead_name != "MLII":
        print(f"  {record_name}: MLII unavailable; using {lead_name}")

    annotation = wfdb.rdann(str(record_base), "atr")
    events = []
    ignored = 0
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        class_id = SYMBOL_TO_CLASS.get(symbol)
        if class_id is None:
            ignored += 1
            continue
        sample = int(sample)
        if 0 <= sample < len(record.p_signal):
            events.append((sample, int(class_id)))

    if ignored:
        print(f"  {record_name}: ignored {ignored} non-beat annotation symbols")
    return _build_record(
        record_name, lead_name, record.p_signal[:, lead_index], int(record.fs), events
    )


def load_record_for_inference(data_dir: Path, record_name: str) -> Tuple[RecordData, float]:
    """Load any supported WFDB sampling rate and resample it to the model's 360 Hz."""
    record_base = Path(data_dir) / record_name
    source_record = wfdb.rdrecord(str(record_base))
    source_fs = float(source_record.fs)
    if not np.isfinite(source_fs) or source_fs <= 0:
        raise ValueError(f"Invalid sampling rate for {record_name}: {source_record.fs}")
    if source_fs <= 80.0:
        raise ValueError(
            f"{record_name} 的源采样率为 {source_fs:g} Hz；当前预处理包含 40 Hz 低通，"
            "因此源采样率必须高于 80 Hz。"
        )

    lead_index, lead_name = _select_lead(source_record)
    if lead_name != "MLII":
        print(f"  {record_name}: MLII unavailable; using {lead_name}")

    annotation = wfdb.rdann(str(record_base), "atr")
    source_events = []
    ignored = 0
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        class_id = SYMBOL_TO_CLASS.get(symbol)
        if class_id is None:
            ignored += 1
            continue
        sample = int(sample)
        if 0 <= sample < len(source_record.p_signal):
            model_sample = int(round(sample * FS / source_fs))
            source_events.append((model_sample, int(class_id)))

    if ignored:
        print(f"  {record_name}: ignored {ignored} non-beat annotation symbols")

    raw_signal = source_record.p_signal[:, lead_index].astype(np.float64, copy=False)
    source_rate = Fraction(str(source_fs)).limit_denominator(100_000)
    ratio = Fraction(FS, 1) / source_rate
    if ratio.numerator != ratio.denominator:
        raw_signal = resample_poly(
            raw_signal, ratio.numerator, ratio.denominator
        )
    print(
        f"  {record_name}: source {source_fs:g} Hz; "
        f"resampled to model input {FS} Hz"
    )
    model_record = _build_record(
        record_name, lead_name, raw_signal, FS, source_events
    )
    return model_record, source_fs

def load_records(data_dir: Path, record_names: Sequence[str]) -> List[RecordData]:
    data_dir = Path(data_dir)
    missing = [
        name for name in record_names
        if not (data_dir / f"{name}.hea").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing record header(s): {', '.join(missing)} in {data_dir}"
        )
    records = []
    for name in record_names:
        print(f"Loading {name} ...")
        records.append(load_record(data_dir, name))
    return records


class EventCenteredWindowDataset(Dataset):
    """One 16-second, pointwise-labeled window centered on each S/V/F/Q beat."""

    def __init__(self, records: Sequence[RecordData]):
        self.records = list(records)
        self.items = []
        for record_index, record in enumerate(self.records):
            for (sample, class_id), (original_sample, _) in zip(
                record.events, record.original_events
            ):
                if class_id in CENTER_CLASSES:
                    self.items.append((record_index, sample, class_id, original_sample))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record_index, center_sample, center_class, _ = self.items[index]
        record = self.records[record_index]
        padded_start = center_sample + WINDOW_HALF
        x = record.signal_padded[padded_start:padded_start + WINDOW_SAMPLES]
        y = record.target_padded[padded_start:padded_start + WINDOW_SAMPLES]

        # Padded targets are background for display, but are excluded from loss.
        source_start = center_sample - WINDOW_HALF
        valid_start = max(0, -source_start)
        valid_end = min(WINDOW_SAMPLES, len(record.signal) - source_start)
        mask = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        if valid_end > valid_start:
            mask[valid_start:valid_end] = 1.0

        return (
            torch.from_numpy(x.copy()).unsqueeze(0),
            torch.from_numpy(y.astype(np.int64, copy=True)),
            torch.from_numpy(mask),
            torch.tensor(center_class, dtype=torch.long),
            torch.tensor(record_index, dtype=torch.long),
            torch.tensor(center_sample, dtype=torch.long),
        )

    def pointwise_support(self, indices: Sequence[int]) -> np.ndarray:
        counts = np.zeros(6, dtype=np.int64)
        for index in indices:
            record_index, center_sample, _, _ = self.items[int(index)]
            record = self.records[record_index]
            source_start = center_sample - WINDOW_HALF
            source_end = source_start + WINDOW_SAMPLES
            start = max(0, source_start)
            end = min(len(record.signal), source_end)
            if end > start:
                counts += np.bincount(
                    record.target[start:end].astype(np.int64), minlength=len(counts)
                )
        return counts

    def center_class_support(self, indices: Sequence[int]) -> np.ndarray:
        counts = np.zeros(6, dtype=np.int64)
        for index in indices:
            _, _, class_id, _ = self.items[int(index)]
            counts[class_id] += 1
        return counts

    def events_for_window(
        self, index: int, original: bool = False
    ) -> List[Tuple[int, int]]:
        record_index, center_sample, _, _ = self.items[index]
        record = self.records[record_index]
        source_start = center_sample - WINDOW_HALF
        source_end = source_start + WINDOW_SAMPLES
        reference_events = record.original_events if original else record.events
        return [
            (sample - source_start, class_id)
            for sample, class_id in reference_events
            if source_start <= sample < source_end
        ]


def load_signal_only(data_dir: Path, record_name: str) -> RecordData:
    """Load and preprocess a record without annotation, for inference."""
    record_base = Path(data_dir) / record_name
    record = wfdb.rdrecord(str(record_base))
    lead_index, lead_name = _select_lead(record)
    return _build_record(
        record_name,
        lead_name,
        record.p_signal[:, lead_index],
        int(record.fs),
        [],
    )
