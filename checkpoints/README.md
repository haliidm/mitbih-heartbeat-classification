# 预训练权重

三个文件随代码仓库发布：ecg_cnn_transformer_16s.pt、ecg_cnn_transformer_32s.pt、ecg_cnn_transformer_64s.pt。

均为现有 MIT-BIH 训练结果的原样副本，没有重新训练或修改张量。内部按窗口划分；不应将 MIT-BIH 全库指标称为独立患者测试结果。ST-T 未用于这些模型的训练。

SHA256 与字节大小见 ../docs/model_manifest.json。历史训练配置见 ../reports/published/ 对应模型目录。


## 权重许可

发布权重采用 [Apache License 2.0](../LICENSE)，允许按条款商用。已移除训练配置中的本机绝对路径，模型张量保持不变。
