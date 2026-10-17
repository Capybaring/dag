# 不规则 DIST 9×9：DAG 与 t-PatchGNN 数据接口

本入口保留 DAG/PatchDelay 的 `[历史 Y, 历史 X]`、已知未来 X、通道注意力、Key/Value 延迟注入和预测头，接入 t-PatchGNN 的观测记录、实际时间 Patch、min-max 归一化和 masked 宏平均评估。原 `scripts/run_benchmark.py` 规则数据入口继续使用；真实不规则数据使用下述新增入口。

## 运行

使用原 requirements.txt 中的 NumPy、pandas、PyTorch、scikit-learn 等依赖，无新增依赖。将前面整理好的 `DIST_irregular_9x9.csv` 放到 `dataset/forecasting/`，然后运行：

```bash
bash scripts/w_future/PatchDelay_DIST_irregular_9x9.sh
```

也可以指定外部文件：

```bash
DATA_PATH=/absolute/path/DIST_irregular_9x9.csv \
  bash scripts/w_future/PatchDelay_DIST_irregular_9x9.sh
```

默认历史 14,400 秒（4 小时）、预测 7,200 秒（2 小时）、Patch 960 秒（16 分钟），非重叠 Patch，delay_context=4 包含当前块及前 3 块。默认 batch_size=2、d_model=64、MSE、Adam(lr=0.001)、最多 1000 轮、验证 MSE patience=10。自动使用 CUDA，未提供 CUDA 时使用 CPU。

首次检查流程：

```bash
EPOCHS=1 BATCH_SIZE=2 \
  bash scripts/w_future/PatchDelay_DIST_irregular_9x9.sh \
  --max-records 12 --d-model 8 --d-ff 16 --te-dim 3 \
  --save-path result/DIST_irregular/smoke
```

`--max-records` 只用于流程检查，会改变划分和缩放所用样本，不可用于正式报告。正式实验移除此参数。关闭 Delay 用 `USE_DELAY=0`。设置 `KEEP_RATE=0.5` 可按官方 subsample_timepoints 的方式，在历史部分按时间点随机保留 50% 观测；未来 X 和未来 Y 标签不删。默认 KEEP_RATE=1，不人为丢弃真实不规则数据。

## 输入与变量角色

支持统一的 `date,data,cols` 或原始 `time,variable,value` CSV。时间列可以是数值相对秒，或规则版日期时间字符串。日期时间统一换算为相对秒；数值时间保留原时间原点。不伪造日期、不做插值或重采样。其它列/非选定变量不参与模型。

默认 9 个 PV 为 Y、同回路 9 个 OP 为 X，回路顺序为 FIC05、FIC09、FIC11、LIC02、LIC03、LIC04、LIC05、LIC06、LIC12。按名称选择，不依赖 CSV 的变量排列顺序或 target_channel 位置。也可以通过 `--targets NAME... --exog NAME...` 指定其它数据集的变量角色。

读取后数据内部按 `[Y, X]` 排列。值与真实观测 mask 分开存储；真实值为 0 时 mask 仍为 True。按相同时间与变量重复的观测取均值，遵循官方 PhysioNet reduce="average" 约定；合并条数写入 manifest。当前 DIST 的 483 条多余重复记录同键同值，取均值不会改变观测含义。原 CSV 不修改。

当前真实文件全量读取结果：时间并集 `(1275114,18)`，合并重复记录 483 条。

## 官方标准与 DIST 的适配边界

参考提交：[`usail-hkust/t-PatchGNN@00c94e7bbaf21c71b03ed84ff690ae59e37129e5`](https://github.com/usail-hkust/t-PatchGNN/tree/00c94e7bbaf21c71b03ed84ff690ae59e37129e5)。采用其 PhysioNet/MIMIC 分支的数据协议。USHCN/Activity 另有数据集专用处理，不存在适用于所有数据集的唯一预处理。

| 环节 | 官方实现 | 本入口 |
|---|---|---|
| 记录 | `(record_id, tt, vals, mask)` | 相同；DIST 连续流先切成不重叠的 6 小时记录 |
| 时间划分 | 记录内 `tt < history` 是输入，其余为预测查询 | 相同；边界时刻属于未来 |
| 数据集划分 | 80% seen/20% test，seen 内 75% train/25% val；random_state=42 | 相同调用；约 60/20/20，保存全部 record_id |
| 数值缩放 | get_data_min_max(seen_data)，只统计真实观测，min-max，常数范围加 1e-8 | 相同；seen 包括训练和验证，不含测试 |
| 时间缩放 | 原相对时间 / seen 中最大相对时间 | 相同；历史和未来使用同一个尺度 |
| Patch | 固定时间宽度、每变量可变观测数量，观测顺序压紧，batch 内 padding | 相同 `[B,K,L,C]` 布局；没有把缺失填成真实观测 |
| 预测标签 | 记录内未来实际查询时间，变长 padding 与 mask | 相同；只监督 Y |
| 训练损失 | masked MSE，先每变量平均再对有效变量平均 | 相同，未沿用旧规则入口的 MAE |
| 验证/测试 | 全部 batch 累加每变量误差和计数，再宏平均；RMSE=sqrt(MSE) | 相同，非 batch 均值的平均 |
| 早停/最优模型 | 以验证 MSE 选择最优轮 | 相同；保存最佳 checkpoint，在最佳轮上评价测试 |
| 未来协变量 | t-PatchGNN 没有 DAG 的未来 X 输入 | 按用户确认的 DAG 已知未来 OP 设定扩展；未来 X 本来的缺测仍受 mask 控制 |

DIST 没有患者/独立个体 ID，因此先按原相对时间从 0 开始切 `[start,start+history+forecast)`，每条记录覆盖 6 小时。最后不足 6 小时的尾段不参与；没有历史或没有未来 Y 标签的记录跳过并报告。当前文件共有 352 条完整候选记录，其中 50 条因缺历史/目标跳过，302 条可用，尾段 20,506 秒未使用。正式默认划分为 180 训练、61 验证、61 测试记录。

这一步是连续工业数据适配，不是官方已经发布的 DIST benchmark。不能把随机重叠滑窗当独立记录再分组。该协议也不是“未来日期留出”的时间外推测试。此前数据说明的时间跨度 70/15/15 划分未用于此入口，以免混用两个标准。

数值缩放严格采用官方 seen=train+validation 参数，而不是训练集单独拟合。测试值可超出 [0,1]，不裁剪。MAPE 同官方 compute_error 默认分支：在归一化目标上排除 0，使用原分母而非绝对分母；因此不将其作为主要结论。MSE/MAE/RMSE 是主要指标，mape 保存为比例。

相对于官方实现的防护：全空 Patch 保留一个 mask=False 的 padding 槽避免零长度张量；整批无标签跳过；全部无有效目标时明确报错；隐藏位置的 NaN 用 where 去除。若 MAPE 完全没有非零有效目标，保存 null。有效数据上的数值与官方函数一致。

## 模型改动范围

- `IrregularPatchEncoder.forward_patches`：复用已有时间编码和 TTCN 权重，接收按实际时间打包的 Patch。原规则 forward 不变。
- `PatchExoModel`：抽取共享预测头，增加按各记录实际查询时刻选择未来 Patch 的 gather；原规则索引路径保留。
- `PatchDelayModel`：把实际查询索引传入原 Key/Value Delay 路径，DelayEncoder 本身不变。
- `IrregularPatchDelayModel`：连接上述接口，预测时不读取 future Y 或 target mask。
- `data/irregular.py` 与 `evaluation/irregular.py`：数据与评估适配。
- `scripts/run_irregular.py`：独立训练入口，避免把规则假设扩散到原 DAG benchmark。

没有替换为 t-PatchGNN 的 Transformer/GNN，没有改变原 DAGModel，没有把当前 Delay 表征解释为显式因果 lag。

## 输出与复现

保存目录默认 `result/DIST_irregular/PatchDelay/`：

- `manifest.json`：参数、变量顺序、缩放参数、time_max、训练/验证/测试 record_id、跳过记录数、重复数、是否 smoke test。
- `best_model.pt`：最佳 state_dict、manifest、最佳轮与验证 MSE。
- `metrics.json`：最佳轮、验证 MSE、测试 MSE/MAE/RMSE/MAPE、各目标有效观测数、逐轮验证日志。

数据和训练产物保持在 .gitignore 范围内，不推送数据文件。重新训练时使用独立 save-path 保存不同实验。

验证命令：

```bash
python -m unittest discover -s tests -v
python tests/check_tpatchgnn_parity.py --reference /path/to/t-PatchGNN
```

第二条直接载入官方仓库的纯函数，对比 Patch 数据/时间/mask、min-max 归一化、未来 padding、masked 宏平均训练误差与全局测试指标，不复制官方实现进本仓库。

## 验证记录（2026-10-17）

- 合成数据测试覆盖两种 CSV、日期转换、真实 0 与缺测、重复取均值、独立记录划分、seen 缩放、物理时间 Patch、每记录查询映射、空输入、梯度、隐藏值不变性、future Y 不泄漏、历史丢弃不删除标签、跨 batch 指标累积。
- 对照原提交 `36b08de` 的规则模型，在单目标及 9×9、相同 state_dict 下逐值一致。
- 官方纯函数数值对照通过（rtol=1e-6, atol=1e-7）。
- 真实 DIST 全量读取后，12 条记录、d_model=8 的两轮训练/验证/测试通过。best epoch=2，验证 MSE=0.07640064，测试 MSE=0.09647167。仅为接口检查，不代表正式精度。
- 默认 d_model=64、batch_size=2，在真实数据 5 条记录上完成一轮训练/验证/测试；验证 MSE=0.05640210，测试 MSE=0.10960540。此项同为流程检查。
- CUDA 和完整 1000 轮实验未验证。
