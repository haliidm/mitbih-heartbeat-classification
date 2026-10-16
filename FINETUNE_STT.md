> 历史说明：旧实验输出已归入 `archive/legacy_20261016.zip`；当前目录与命令见 [项目入口](docs/项目入口.md)。

# ST-T微调入口

在 `项目根目录` 中运行。默认使用现有MIT-BIH模型，输出到新目录 `runs/stt_finetune`。

```powershell
python -m mitbih_aami.legacy_finetune_stt --prepare-only
python -m mitbih_aami.legacy_finetune_stt --epochs 10 --learning-rate 1e-5 --patience 3
```

第一条命令只统计ATR心搏类别并保存患者分组；第二条才开始训练。支持 `--checkpoint`、`--database-dir`、`--output-dir`、`--batch-size`（默认16）和 `--device auto/cpu/cuda`。显存不足时减小batch-size。

## 数据划分

依据[PhysioNet公布的同患者记录关系](https://physionet.org/content/edb/1.0.0/)绑定：e0118–e0122、e0123–e0126、e0129/e0133、e0136/e0139、e0147/e0148、e0154/e0155、e0162/e0163；其余记录分别作为独立组。绑定关系只应用于实际存在的记录；当前90条记录中没有e0120，最终得到79个患者组，与官方描述一致。

以组为单位约6:2:2划分，固定随机种子42。在5000个候选分组中，仅依据参考类别数量选择较均衡的分配，不根据模型效果选择划分。优先训练类别覆盖，并尽量让分布在至少3个患者组的类别覆盖三组；不能保证少见类别在三组均充足。缺失类别会打印提示。

`split_manifest.json`保存固定划分，后续重复使用。`record_class_counts.csv`列出每条记录的组、集合和N/S/V/F/Q数量。可用 `--split-file` 导入相同结构的划分；校验记录无重复、完整覆盖数据库，以及已知同患者记录不跨集合。已有划分不会被另一份划分覆盖。这里读取测试集ATR用于划分统计，但不加载测试信号、不利用测试预测或测试指标选模。

全库结果之前已经被查看，因此应披露这一背景；该测试集合是本次微调保留集合，并非从未接触过的外部数据库。

## 训练与验证

- 复用app加载流程：按源采样率映射到360Hz，完整记录0.67–40Hz滤波、逐记录z-score、参考峰校正、生成约360ms参考标签，再切窗。标签由ATR生成，不由V4预测生成。
- 训练只生成S/V/F/Q中心的16秒窗口，不添加自然滑窗或N中心窗口。窗口内背景和N/S/V/F/Q逐点标签全部保留，全部有效点参与损失；记录边界处填充点不计损失。`--minority-repeats 1`为默认，每个中心心搏生成一个窗口；2重复两份中心窗口，0不允许。当前固定划分共有618+2618+242+8=3486个中心窗口（默认不重复），相邻窗口允许重叠，但不跨患者集合。只有N心搏的训练记录不会贡献窗口。
- 全网络微调，加载已有 `model_state`，重新初始化AdamW。损失为16秒窗口所有有效点的交叉熵；默认背景权重0.15、其他类别1，未叠加逆频率类别权重。
- 验证集每轮均进行完整记录16秒窗、8秒步长推理并融合，不对验证集过采样。验证日志保存逐点混淆矩阵、位置匹配后5类混淆矩阵和固定V4规则下的位置/分类指标。
- 默认按验证集有参考支持的N/S/V/F类别宏平均F1选模，同时打印R峰F1。可用 `--selection-classes N S V F Q`更改目标。Q默认不参与选模，但仍输出Q指标；无参考支持的类别不参与宏平均，并保存实际参与类别。
- 第0轮先验证原模型；只有超过原模型/当前最佳指标才更新最佳权重。连续3轮没有改善早停。`best_model.pt`的epoch为0时表示微调未超过初始模型。
- 保存 `finetune_config.json`、`training_history.json`、`baseline_validation.json`、`training_windows.json`、训练标注校正审计和最终摘要。完整记录验证比随机窗口验证耗时更长。

训练不自动使用测试集。完成后单独运行：

```powershell
python -m mitbih_aami.legacy_finetune_stt --evaluate-test
```

此命令仅评估：读取本次保存的原模型路径与最佳模型，调用V4评估器分别对固定测试记录生成 `test_pretrained_v4` 和 `test_finetuned_v4` 两套报告。两次评估使用相同数据预处理、参考峰校正、100ms匹配容差及V4规则。已有评估报告在重复评估时更新。若指定过其他output-dir/database-dir，此处也传入相同目录。

查看新模型：

```powershell
python -m mitbih_aami.desktop_app --checkpoint .\runs\stt_finetune\best_model.pt
```

训练入口不覆盖已有微调运行。重新试验使用新的 `--output-dir`，并通过 `--split-file .\runs\stt_finetune\split_manifest.json`保持相同划分。原MIT-BIH训练入口及权重保持独立。
