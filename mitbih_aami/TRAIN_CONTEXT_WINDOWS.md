# MIT-BIH 32 秒 / 64 秒上下文训练

本实验沿用已有 MIT-BIH 模型的卷积多尺度 + 双向 Transformer 架构、360 Hz 预处理、全记录 R 波矫正和逐点标签。训练样本仍由 S/V/F/Q 中心窗口与 AFIB/AFL 节律背景下补充的 N 窗口构成，沿用相同随机种子及 7:1:2 窗口划分；只把输入上下文从 16 秒扩展到 32 秒或 64 秒。输出标签与输入一样长，仍按每个采样点监督。

为控制对比变量，类别损失权重沿用同一批训练窗口按原 16 秒设置统计出的权重；S 窗口抽样因子、AFIB/AFL N 窗口补充数、学习率和随机种子保持一致。模型根据窗口长度自动配置 Transformer 位置编码。64 秒输入的注意力显存开销明显更高，默认 32 秒和 64 秒的 batch size 均为 8；整条记录推理的默认批量也会按长度自动降低。

## 训练命令（PowerShell）

在 `项目根目录` 项目根目录运行：

```powershell
python -m mitbih_aami.train_long_context `
  --database-dir .\mit-bih-arrhythmia-database-1.0.0 `
  --output-dir .\runs\mitbih_32s `
  --window-seconds 32 `
  --epochs 50
```

```powershell
python -m mitbih_aami.train_long_context `
  --database-dir .\mit-bih-arrhythmia-database-1.0.0 `
  --output-dir .\runs\mitbih_64s `
  --window-seconds 64 `
  --epochs 50
```

可使用 `--batch-size` 参数调整训练批量。两次必须使用不同的输出目录。每次训练保存最佳模型、训练历史、配置、划分清单和逐点分类测试指标。

## 完整记录评估

分别使用各自模型运行相同的 V5 完整记录评估：

```powershell
python -m mitbih_aami.evaluate_records `
  --database-dir .\mit-bih-arrhythmia-database-1.0.0 `
  --checkpoint .\checkpoints\ecg_cnn_transformer_32s.pt `
  --output-dir .\runs\mitbih_32s_eval_v5
```

```powershell
python -m mitbih_aami.evaluate_records `
  --database-dir .\mit-bih-arrhythmia-database-1.0.0 `
  --checkpoint .\checkpoints\ecg_cnn_transformer_64s.pt `
  --output-dir .\runs\mitbih_64s_eval_v5
```

该完整库评估便于与现有报告的规则五指标直接对照，但原项目将窗口随机划分，训练/验证/测试可来自相同记录，所以全库结果不能视为患者独立的最终泛化结果。对照时同时查看训练入口输出的 held-out 窗口逐点指标，以及全库事件报告；不要只根据全库分数下结论。
