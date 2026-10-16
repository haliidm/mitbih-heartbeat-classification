# 补充房颤/房扑背景N窗口的MIT-BIH训练

新入口：`mitbih_aami.train_16s`。从随机权重训练，不使用ST-T，原有脚本、权重与报告保留。

保留原有16秒S/V/F/Q中心窗口及其seed=42的7:1:2随机划分，S抽样权重默认2。读取全库ATR辅助节律标注，将AFIB/AFL标注持续到下一条节律标注（最后一段到记录末尾）。仅在完整16秒都位于相应节律内，且窗口内参考心搏全部为N时补充窗口；所有逐点标签沿用原数据生成流程，绝不将房颤/房扑改为S。

候选窗口中心对齐到最近的参考N心搏；排除与任意原始窗口相交者，并避免补充窗口互相重叠。在每条记录、每种节律的合格候选中按时间均匀选择，默认最多6个。通过`--rhythm-n-per-record`统一调整，不针对某条出错记录设置。

补充窗口在各记录/节律内随机分配训练、验证、测试；数量不足3时优先训练及验证，因此新增窗口比例不是严格7:1:2。原始窗口划分完全保留。新增窗口虽然没有时间交叉，但原有窗口随机划分仍可能存在重叠，整个实验不是患者独立测试。

默认实际选出31个窗口，来自7条记录：201、202、203、210、219、221、222。具体记录、数量与时间见`runs/mitbih_rhythm_preview/rhythm_n_windows.json`，其中训练20、验证6、测试5。该补充量很小，不保证一次训练即可解决误报。

训练：

```powershell
python -m mitbih_aami.train_16s --epochs 50 --patience 50 --output-dir .\runs\mitbih_rhythm_n
```

仅生成预览（使用一个全新的输出目录）：

```powershell
python -m mitbih_aami.train_16s --prepare-only --output-dir .\runs\mitbih_rhythm_preview_new
```

训练日志额外报告补充验证窗口中参考N采样点被预测为S的比例；这是逐点指标，不是心搏级误报率。仍按MIT-BIH全部验证窗口逐点宏平均F1选模。

V5全库检查：

```powershell
python -m mitbih_aami.evaluate_records --database-dir .\mit-bih-arrhythmia-database-1.0.0 --checkpoint .\checkpoints\ecg_cnn_transformer_16s.pt --output-dir .\runs\mitbih_rhythm_n\mitbih_full_v5 --tolerance-ms 100
```

app：

```powershell
python -m mitbih_aami.desktop_app --checkpoint .\checkpoints\ecg_cnn_transformer_16s.pt --postprocess v5
```

训练完成后的全库报告包含补充过的训练片段，应结合保留的验证/测试补充窗口及外部评估解读。不要把已针对分析和训练的202片段当作未知泛化样本。
