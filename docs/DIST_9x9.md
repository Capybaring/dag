# DIST：9 个 PV 目标与 9 个 OP 辅助输入

此任务同时预测 FIC05、FIC09、FIC11、LIC02、LIC03、LIC04、LIC05、LIC06、LIC12 九个回路的 PV，使用对应的九个 OP 作为辅助输入。DIST 原有另外 19 个 FI/TI/PI 测量列不参与本任务。

这里的 target/exog 是预测任务角色。OP 可能由反馈控制产生，未来值不一定提前已知，因此配置使用 `infer_use_future=0`。这不会证明统计外生性或控制动作的因果效应。

## 数据文件

转换后文件为 `dataset/forecasting/DIST_9x9.csv`，格式为变量块连续存放的 `date,data,cols` 长表。每个变量块包含整段相同时间轴，不能将行改成按时间点交错排列。

| CSV 通道索引（从 0 开始） | 变量 | 任务角色 |
|---:|---|---|
| 0–8 | FIC05.OP、FIC09.OP、FIC11.OP、LIC02.OP、LIC03.OP、LIC04.OP、LIC05.OP、LIC06.OP、LIC12.OP | 辅助输入 |
| 9–17 | FIC05.PV、FIC09.PV、FIC11.PV、LIC02.PV、LIC03.PV、LIC04.PV、LIC05.PV、LIC06.PV、LIC12.PV | 九个预测目标 |

变量保留原名称，不将九个目标统一改成同名 OT。原始 DIST 有 133,920 个时间点、固定 20 秒间隔；完整输出共 2,410,560 行长表记录、18 个变量。

转换不插值、不重采样、不提前标准化，只选择变量、排列变量块并将 Unix 时间戳转为日期字符串。暂沿用 Asia/Shanghai 时区解释，保留毫秒。若实际时区不同，可通过 `--timezone` 明确指定。转换不修改 NPZ，已有 CSV 会拒绝覆盖。

数据在 `.gitignore` 排除的 dataset 目录，本提交只提供转换代码、实验配置、修正和测试，不包含私有原始或转换数据。

## 准备与运行

在仓库根目录、已安装仓库依赖的 Python 环境中运行：

```sh
python scripts/prepare_dist_9x9.py --input /你的路径/DIST_DataSet.npz
python scripts/run_dist_9x9.py
```

如果已拿到转换好的 CSV，直接复制到 `dataset/forecasting/DIST_9x9.csv` 即可。保留已有 FORECAST_META.csv；如果目录缺少该文件，需要运行转换器生成元数据或复制随数据提供的元数据文件。仓库会自动注册新 CSV。

转换器写出 `DIST_9x9.validation.json`，其中包含变量来源索引、输出索引、原始和输出 SHA-256、全部时间块与数值核对、仓库原始读取器核对和九目标/九辅助拆分检查。

查看配置与启动命令但不开始训练：

```sh
python scripts/run_dist_9x9.py --dry-run
```

先用真实数据的前 2,400 个时间点验证训练流程：

```sh
python scripts/prepare_dist_9x9.py --input /你的路径/DIST_DataSet.npz --output dataset/forecasting/DIST_9x9_smoke.csv --limit 2400
python scripts/run_dist_9x9.py --data-name DIST_9x9_smoke.csv --num-epochs 1 --num-rollings 4
```

这段运行验证代码路径，不用于报告正式预测精度。全量训练仍需另行执行。

## 调整位置

1. `config/dist_9x9_config.json` 的 `evaluation_config.strategy_args.target_channel` 为 `[9,10,11,12,13,14,15,16,17]`，同时选中九个 PV。不能沿用 `[-1]`，否则只预测 LIC12.PV。
2. 同一配置中的 `model_hyper_params.infer_use_future=0`，在验证/推理的 eval 模式中不使用实际未来 OP。模型内部由历史估计未来 OP；训练阶段仍使用已观测未来 OP 做辅助学习。
3. `series_dim=9`、`enc_in=18` 和输出维度由原框架根据目标与辅助输入自动确定，不需要把模型输出层手工改成固定 9，也不需要改全局目标默认值。
4. `seq_len=168` 是历史 56 分钟；`horizon=24` 是未来 8 分钟。默认 50 个训练轮次，最多 48,000 个滚动预测起点，可通过启动器参数修改。参数只是起始配置，未针对本任务调优。
5. 数据按时间顺序 70%/10%/20% 划分：93,744/13,392/26,784 点。标准化参数由模型在训练段拟合。

默认报告指标对九个目标和预测窗口进行汇总，不是九份逐变量指标。若需要逐变量比较，应另行保存预测结果并按通道计算指标。

## 多目标注意力修正

`ts_benchmark/baselines/dag/layers/TC_EncDec.py` 中，目标 patch 按“样本、目标通道”顺序展开。原先 `attn_alpha.repeat(series_dim, ...)` 按另一种顺序重复样本门控值，多目标批次会将部分通道应用到其他样本的门控。改为 `repeat_interleave(series_dim, dim=0)`，让每个样本的九个目标共用该样本自己的门控。单目标行为不变。

`tests/test_dist_9x9.py` 覆盖：乱序源索引仍按变量名选择、完整时间/数值回读、九目标拆分、批量预测与逐样本预测一致、九目标损失反向传播，以及 `infer_use_future=0` 不依赖提供的未来 OP。

```sh
python -m pytest tests/test_dist_9x9.py -q
```

测试工具 pytest 需要另行安装。正常训练依赖仍按仓库 requirements.txt；无需为转换器安装 PyTorch。

## 本次验证范围

完整 133,920 点转换数据通过逐值回读、时间轴和原仓库读取器检查。上述 5 项测试全部通过；修复前，多目标批量/逐样本一致性测试失败，修复后通过。

真实数据前 2,400 点已完成 1 轮训练和 4 个滚动预测起点，七个导出指标均为有限数值，运行日志没有错误。这仅验证训练、预测和报告流程；未执行全量 50 轮训练，不能据此评价正式预测精度。

本次本地验证使用独立 Python 3.12 环境，核心版本为 PyTorch 2.4.1、NumPy 1.26.4、pandas 2.2.3；为适配 Python 3.12，验证环境使用 Ray 2.31.0。未修改仓库的依赖版本文件。
