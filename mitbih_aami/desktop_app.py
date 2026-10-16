"""Desktop viewer for reference and model-predicted ECG pointwise labels."""

import argparse
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .label_config import CLASS_NAMES, FS, LABEL_RADIUS_MS, WINDOW_SECONDS, WINDOW_SAMPLES
from .ecg_data import load_record_for_inference
from .record_inference_metrics import predict_record
from .cnn_transformer import MultiScaleCNNTransformer
from .legacy_beat_postprocess_v4 import refine_pointwise_labels_v4, color_final_regions_v4
from .beat_postprocess import display_labels_v5


CLASS_LABELS_ZH = ("背景", "正常心搏", "室上性早搏", "室性早搏", "融合心搏", "未分类心搏")
CLASS_COLORS = ("#f3f3f3", "#4e79a7", "#edc949", "#e15759", "#76b7b2", "#af7aa1")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/ecg_cnn_transformer_16s.pt"
def configure_chinese_font():
    """Choose an installed CJK font so Matplotlib can render Chinese labels."""
    preferred_names = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "DengXian",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    )
    installed = {
        entry.name.casefold(): entry.name
        for entry in font_manager.fontManager.ttflist
    }
    selected = next(
        (installed[name.casefold()] for name in preferred_names
         if name.casefold() in installed),
        None,
    )
    if selected:
        rcParams["font.family"] = [selected]
        rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return selected


CHINESE_FONT = configure_chinese_font()


class PointwiseLabelViewer:
    def __init__(self, root: tk.Tk, checkpoint: Path, postprocess="v5"):
        self.postprocess = postprocess
        self.split_boundaries = []
        self.split_lines = []
        self.root = root
        self.root.title("ECG 心搏标签浏览器")
        self.root.geometry("1280x940")
        self.root.minsize(900, 650)
        if CHINESE_FONT:
            ttk.Style(self.root).configure(".", font=(CHINESE_FONT, 9))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint
        self.hea_path: Path | None = None
        self.record = None
        self.predicted_labels: np.ndarray | None = None
        self.corrected_labels: np.ndarray | None = None
        self.result_queue: queue.Queue = queue.Queue()
        self.busy = False
        self.signal_limit = 1.0

        self.checkpoint_text = tk.StringVar(value=str(checkpoint))
        self.record_text = tk.StringVar(value="尚未导入记录")
        self.status_text = tk.StringVar(value=f"设备：{'GPU' if self.device.type == 'cuda' else 'CPU'}")
        self.window_text = tk.StringVar(value="浏览窗口：—")
        self.slider_value = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.after(100, self._poll_result_queue)

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=(10, 8))
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls, text="选择模型文件", command=self._choose_checkpoint).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(controls, textvariable=self.checkpoint_text, width=76).grid(row=0, column=1, sticky="w")
        ttk.Button(controls, text="选择 WFDB hea 文件", command=self._choose_record).grid(row=1, column=0, padx=(0, 8), pady=(8, 0))
        ttk.Label(controls, textvariable=self.record_text).grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.run_button = ttk.Button(controls, text="导入并推理整条记录", command=self._start_inference)
        self.run_button.grid(row=0, column=2, rowspan=2, padx=(12, 0), sticky="ns")
        ttk.Label(
            controls,
            text="读取同名 dat 和 atr；输入信号会重采样到模型要求的 360 Hz。",
        ).grid(row=2, column=1, columnspan=2, sticky="w", pady=(7, 0))
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(12, 8.2), dpi=100)
        self.ax_reference, self.ax_prediction, self.ax_corrected = self.figure.subplots(3, 1, sharex=True)
        self.figure.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.13, hspace=0.42)
        self.colormap = ListedColormap(CLASS_COLORS)
        self.color_norm = BoundaryNorm(np.arange(-0.5, len(CLASS_LABELS_ZH) + 0.5, 1), self.colormap.N)

        blank = np.zeros((1, WINDOW_SAMPLES), dtype=np.uint8)
        initial_extent = (0, WINDOW_SECONDS, -1, 1)
        self.reference_image = self.ax_reference.imshow(
            blank, aspect="auto", interpolation="nearest", origin="lower",
            extent=initial_extent, cmap=self.colormap, norm=self.color_norm,
            alpha=0.20, zorder=0,
        )
        self.prediction_image = self.ax_prediction.imshow(
            blank, aspect="auto", interpolation="nearest", origin="lower",
            extent=initial_extent, cmap=self.colormap, norm=self.color_norm,
            alpha=0.20, zorder=0,
        )
        self.corrected_image = self.ax_corrected.imshow(
            blank, aspect="auto", interpolation="nearest", origin="lower",
            extent=initial_extent, cmap=self.colormap, norm=self.color_norm,
            alpha=0.20, zorder=0,
        )
        self.signal_lines = [
            ax.plot([], [], color="#263238", linewidth=0.8, zorder=3)[0]
            for ax in (self.ax_reference, self.ax_prediction, self.ax_corrected)
        ]
        self.ax_reference.set_title("心电信号与参考心搏标签", fontsize=12)
        self.ax_prediction.set_title("心电信号与模型原始推理标签", fontsize=12)
        self.ax_corrected.set_title("心电信号与" + ("第五版" if self.postprocess == "v5" else "第四版") + "规则修正标签", fontsize=12)
        for ax in (self.ax_reference, self.ax_prediction, self.ax_corrected):
            ax.set_ylabel("标准化幅值")
            ax.set_xlim(0, WINDOW_SECONDS)
            ax.set_ylim(-1, 1)
            ax.grid(axis="y", color="#777777", alpha=0.20, linewidth=0.5)
        self.ax_corrected.set_xlabel("记录时间（秒）")
        legend_handles = [
            Line2D([0], [0], color="#263238", linewidth=1.0, label="心电信号"),
            *[
                Patch(facecolor=color, edgecolor="#777777", label=label)
                for color, label in zip(CLASS_COLORS, CLASS_LABELS_ZH)
            ],
        ]
        self.figure.legend(
            handles=legend_handles, loc="lower center", ncol=7,
            frameon=False, bbox_to_anchor=(0.56, 0.02), title="波形与标签",
        )
        plot_frame = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        navigation = ttk.Frame(self.root, padding=(12, 5, 12, 0))
        navigation.pack(side=tk.TOP, fill=tk.X)
        self.previous_window_button = ttk.Button(
            navigation, text="上一个窗口（前移10秒）",
            command=lambda: self._move_window(-10.0), state=tk.DISABLED,
        )
        self.previous_window_button.pack(side=tk.LEFT, padx=(0, 6))
        self.next_window_button = ttk.Button(
            navigation, text="下一个窗口（后移10秒）",
            command=lambda: self._move_window(10.0), state=tk.DISABLED,
        )
        self.next_window_button.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(navigation, text="滑动浏览").pack(side=tk.LEFT, padx=(0, 10))
        self.slider = tk.Scale(
            navigation, from_=0, to=0, resolution=0.1, orient=tk.HORIZONTAL,
            showvalue=False, variable=self.slider_value, command=self._on_slider,
            state=tk.DISABLED,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(navigation, textvariable=self.window_text, width=28, anchor="e").pack(side=tk.LEFT, padx=(10, 0))

        footer = ttk.Frame(self.root, padding=(12, 7, 12, 10))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(footer, textvariable=self.status_text).pack(side=tk.LEFT)
        ttk.Label(
            footer,
            text="三图共用心电信号；底色表示类别。窗口固定16秒。",
        ).pack(side=tk.RIGHT)

    def _choose_checkpoint(self):
        selected = filedialog.askopenfilename(
            title="选择训练完成的模型",
            initialdir=str(self.checkpoint_path.parent) if self.checkpoint_path.parent.exists() else None,
            filetypes=(("模型文件", "*.pt *.pth"), ("所有文件", "*.*")),
        )
        if selected:
            self.checkpoint_path = Path(selected)
            self.checkpoint_text.set(str(self.checkpoint_path))

    def _choose_record(self):
        selected = filedialog.askopenfilename(
            title="选择 WFDB hea 文件",
            filetypes=(("WFDB 头文件", "*.hea"), ("所有文件", "*.*")),
        )
        if selected:
            self.hea_path = Path(selected)
            base = self.hea_path.with_suffix("")
            self.record_text.set(f"{base.name}.hea、{base.name}.dat、{base.name}.atr")
            self.status_text.set("已选择记录；准备检查配套文件。")

    def _start_inference(self):
        if self.busy:
            return
        if self.hea_path is None:
            messagebox.showinfo("需要选择记录", "请先选择该记录的 .hea 文件。")
            return
        if not self.checkpoint_path.is_file():
            messagebox.showerror("找不到模型", f"模型文件不存在：\n{self.checkpoint_path}")
            return
        base = self.hea_path.with_suffix("")
        companions = (base.with_suffix(".dat"), base.with_suffix(".atr"))
        missing = [path.name for path in companions if not path.is_file()]
        if missing:
            messagebox.showerror(
                "配套文件缺失",
                "未找到与 hea 同名的文件：" + "、".join(missing) + "。请把 dat、atr、hea 放在同一文件夹。",
            )
            return

        self.busy = True
        self.run_button.configure(state=tk.DISABLED)
        self.status_text.set("正在读取记录并推理整条信号，请稍候……")
        worker = threading.Thread(
            target=self._load_and_infer,
            args=(self.hea_path, self.checkpoint_path),
            daemon=True,
        )
        worker.start()

    def _load_and_infer(self, hea_path: Path, checkpoint_path: Path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_radius = checkpoint.get("label_radius_ms")
            if checkpoint_radius is not None and int(checkpoint_radius) != LABEL_RADIUS_MS:
                raise ValueError(
                    f"模型标签半径为 {checkpoint_radius} ms，但当前参考标签配置为 "
                    f"±{LABEL_RADIUS_MS} ms。请选用 360 ms 标签训练出的模型。"
                )
            input_samples = int(checkpoint.get("input_samples", WINDOW_SAMPLES))
            model = MultiScaleCNNTransformer(input_samples=input_samples).to(self.device)
            model.load_state_dict(checkpoint["model_state"])
            record, source_fs = load_record_for_inference(hea_path.parent, hea_path.stem)
            logits = predict_record(model, record, self.device)
            predicted = np.argmax(logits, axis=0).astype(np.uint8)
            if self.postprocess == "v5":
                corrected, details = display_labels_v5(predicted, record.signal, sample_rate=FS)
            else:
                corrected, details = refine_pointwise_labels_v4(predicted, sample_rate=FS)
                corrected = color_final_regions_v4(corrected, sample_rate=FS)
            self.result_queue.put(("success", record, predicted, corrected,
                                   checkpoint.get("epoch"), source_fs, details))
        except Exception as exc:
            self.result_queue.put(("error", exc))

    def _poll_result_queue(self):
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result_queue)
            return

        self.busy = False
        self.run_button.configure(state=tk.NORMAL)
        if result[0] == "error":
            self.status_text.set("读取或推理失败。")
            messagebox.showerror("推理失败", str(result[1]))
        else:
            _, self.record, self.predicted_labels, self.corrected_labels, epoch, source_fs, details = result
            self.split_boundaries = []
            for item in details.get("long_regions", []):
                if item["split"]:
                    peaks = np.rint(np.asarray(item["candidate_peaks_seconds"]) * FS).astype(np.int64)
                    self.split_boundaries.extend((((peaks[:-1]+peaks[1:])//2+1)/FS).tolist())
            self.signal_limit = max(float(np.max(np.abs(self.record.signal))) * 1.05, 1.0)
            duration = len(self.record.signal) / FS
            max_start = max(0.0, duration - WINDOW_SECONDS)
            self.slider.configure(to=max_start, state=tk.NORMAL)
            self.slider_value.set(0.0)
            self._draw_window(0.0)
            model_epoch = f"；模型来自第 {epoch} 轮" if epoch is not None else ""
            self.status_text.set(
                f"已完成整条记录推理：{self.record.name}，导联 {self.record.lead_name}，"
                f"原始采样率 {source_fs:g} Hz，模型输入采样率 {FS} Hz，"
                f"设备 {'GPU' if self.device.type == 'cuda' else 'CPU'}{model_epoch}；"
                f"{'第五版' if self.postprocess == 'v5' else '第四版'}额外删除 {len(details['removed_regions'])} 个短区域。"
            )
        self.root.after(100, self._poll_result_queue)

    def _move_window(self, offset_seconds):
        if self.record is None or self.predicted_labels is None:
            return
        max_start = max(0.0, (len(self.record.signal) - WINDOW_SAMPLES) / FS)
        current = getattr(self, "_window_start_seconds", 0.0)
        target = min(max(0.0, current + offset_seconds), max_start)
        self.slider_value.set(target)
        self._draw_window(target)

    def _on_slider(self, value):
        if self.record is not None and self.predicted_labels is not None:
            self._draw_window(float(value))

    def _draw_window(self, requested_start_seconds: float):
        total_samples = len(self.record.signal)
        max_start = max(0, total_samples - WINDOW_SAMPLES)
        start_sample = min(max(0, int(round(requested_start_seconds * FS))), max_start)
        end_sample = min(start_sample + WINDOW_SAMPLES, total_samples)

        reference = self.record.target[start_sample:end_sample]
        prediction = self.predicted_labels[start_sample:end_sample]
        corrected = self.corrected_labels[start_sample:end_sample]
        signal = self.record.signal[start_sample:end_sample]
        missing = WINDOW_SAMPLES - len(reference)
        if missing > 0:
            reference = np.pad(reference, (0, missing), constant_values=0)
            prediction = np.pad(prediction, (0, missing), constant_values=0)
            corrected = np.pad(corrected, (0, missing), constant_values=0)
            signal = np.pad(signal, (0, missing), constant_values=np.nan)

        start_seconds = start_sample / FS
        self._window_start_seconds = start_seconds
        self.previous_window_button.configure(state=tk.NORMAL if start_sample > 0 else tk.DISABLED)
        self.next_window_button.configure(state=tk.NORMAL if start_sample < max_start else tk.DISABLED)
        end_seconds = start_seconds + WINDOW_SECONDS
        time_axis = start_seconds + np.arange(WINDOW_SAMPLES) / FS
        self.signal_limit = max(float(np.max(np.abs(signal))) * 1.05, 1e-3)
        extent = (start_seconds, end_seconds, -self.signal_limit, self.signal_limit)

        for ax, image, labels, line in (
            (self.ax_reference, self.reference_image, reference, self.signal_lines[0]),
            (self.ax_prediction, self.prediction_image, prediction, self.signal_lines[1]),
            (self.ax_corrected, self.corrected_image, corrected, self.signal_lines[2]),
        ):
            image.set_data(labels[np.newaxis, :])
            image.set_extent(extent)
            line.set_data(time_axis, signal)
            ax.set_xlim(start_seconds, end_seconds)
            ax.set_ylim(-self.signal_limit, self.signal_limit)
            ax.set_yticks(np.linspace(-self.signal_limit, self.signal_limit, 5))

        self.ax_reference.set_xticks(np.arange(np.ceil(start_seconds / 2) * 2, end_seconds + 0.01, 2))
        self.ax_prediction.set_xticks(np.arange(np.ceil(start_seconds / 2) * 2, end_seconds + 0.01, 2))
        for line in self.split_lines:
            line.remove()
        self.split_lines = [self.ax_corrected.axvline(t, color="#555555", linestyle="--", linewidth=0.7, alpha=0.8)
                            for t in self.split_boundaries if start_seconds <= t < end_seconds]
        self.window_text.set(f"{start_seconds:.1f}–{end_seconds:.1f} 秒")
        self.canvas.draw_idle()



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--postprocess", choices=("v4", "v5"), default="v5", help="第三图后处理版本；V5灰色虚线表示长区域拆分边界")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    PointwiseLabelViewer(root, args.checkpoint, args.postprocess)
    root.mainloop()


if __name__ == "__main__":
    main()
