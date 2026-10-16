> 历史说明：旧实验输出已归入 `archive/legacy_20261016.zip`；当前目录与命令见 [项目入口](docs/项目入口.md)。

# MIT-BIH从头重训（S窗口采样增强）

只加载完整48条MIT-BIH记录，从随机初始化训练；不加载ST-T微调模型，不读取ST-T数据。
保留原模型、预处理、参考峰校正、约360ms标签及16秒S/V/F/Q中心窗口，窗口中N及背景点仍参与损失。
保留原seed=42与窗口随机7:1:2划分，以便比较；这不是患者独立划分，相邻重叠窗口可能跨内部集合，内部指标不能代表外部泛化。

唯一默认训练策略变化：训练集S中心窗口抽样权重为其他窗口的2倍（可调），带放回抽样，每轮总窗口数不变。该因子是实验起点，不能保证ST-T改善。原按训练标签频数计算的类别损失权重保持不变。验证/测试不重复采样；最佳模型仍只按MIT-BIH验证逐点N/S/V/F/Q宏平均F1选择，日志保存每轮S精确率、召回率和F1。

在项目根目录运行：

```powershell
python -m mitbih_aami.legacy_train_s_weighted --epochs 50 --patience 50 --s-sampling-factor 2 --output-dir .\runs\mitbih_s_retrain
```

若只想完全复现原来的抽样方式，使用`--s-sampling-factor 1`，并指定另一个全新输出目录。不得通过反复查看ST-T结果选择采样因子或最佳轮次。

完成后对ST-T外部评估：

```powershell
python -m mitbih_aami.legacy_evaluate_records_v4 --database-dir .\european-st-t-database-1.0.0 --checkpoint .\runs\mitbih_s_retrain\best_model.pt --output-dir .\runs\mitbih_s_retrain\stt_external_v4 --tolerance-ms 100
```

app查看：

```powershell
python -m mitbih_aami.desktop_app --checkpoint .\runs\mitbih_s_retrain\best_model.pt
```

旧模型MIT-BIH窗口测试S逐点F1约98.86%，不等同于心搏位置和类别联合F1。ST-T性能不佳不能单凭此归因为S未学好；导联/数据库差异和后处理判类也可能影响结果。V4中N占比大于10%规则保持固定。
ST-T此前已被查看且参与过另一个微调实验，应披露这一背景；本次新模型不使用ST-T训练，但这不等同于从未接触过的盲测外部集。
