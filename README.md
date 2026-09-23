# 多模态情感建模

此仓库实现执行 Spec 中的问题2/3核心数值流程，并为问题1提供可复核的时间窗聚合接口。代码只使用本题提供的特征和标签；不把 `raw_text` 或 `text_bert` 当作额外预测输入。

当前版本面向**本地合成数据验证**。没有真实数据时不会启动训练，也不会生成虚构成绩。官方 Pickle 只应从可信的赛题来源读取；数据适配器要求显式开启 Pickle 读取。

## 安装与检查

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

本机已验证运行时为 Python 3.12 + NumPy + PyTorch。目标复现实验环境仍需在 Linux x86_64 / Python 3.10 上重新锁定实际可用版本。

## 数据接口

问题2/3使用 `aligned_50.pkl`，序列位置数为50，维度为 T=768、A=74、V=35。模型入口接收三组 `[batch, time, dimension]` 数组、有效位置 `P`、原始观测 `O0` 与当次可见掩码 `O`。三者语义不同，人工遮挡不改变 `P` 或 `O0`。

训练/验证/测试划分保持官方划分。归一化统计只从 train 的有效且原本可观测位置计算。附件3/4不得用于模型选择。

## 模块

- `data.py`：特征适配、掩码审计与 train-only 标准化。
- `corruption.py`：独立随机源的连续缺失生成和固定评价掩码。
- `models.py`：B0–B5、M0/M1 预测器。
- `metrics.py`、`training.py`、`evaluation.py`：训练、早停与固定场景评价。
- `explanation.py`：遮挡敏感度和证据窗口验证。
- `q1_alignment.py`：将已提取的词/声学/视觉时间记录聚合到50个时间窗；外部特征提取工具尚未集成验证。
- `export.py`：预测 CSV 合同与完整性检查。

## 尚未完成的外部验证

需要收到附件后检查实际字段、长度掩码、标签映射、Q4 位置到真实时间映射以及 Pickle 结构。FFmpeg、OpenFace、openSMILE、BERT/CTC 权重和 Linux 目标环境也需要真实运行验证。源码的合成测试不等于实验训练或赛题效果验证。
