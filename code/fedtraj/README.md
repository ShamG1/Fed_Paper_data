# fedtraj 服务器运行说明

## 0. Fed-SSBR（当前论文，2026-10-22 起）

本次投稿继续 Fed-SSBR（客户端层 CQS + 服务器层 UDR），主实验迁到这套管线：原始 NGSIM 和 highD、CV 锚定骨干、物理空间损失。FedTR 留作下一篇，第 1 节起关于 FedTR 的说明和阶段保持不变。两者共用的运行（同一设置、同一种子的 central、fedavg 等）只训练一次。

### 0.1 配置

| 论文中的名称 | 参数 |
|---|---|
| CQS（客户端层） | `--algo cqs` |
| UDR 单独使用 | `--algo fedavg --reweight udr` |
| Fed-SSBR（CQS+UDR） | `--algo cqs --reweight udr` |
| CQS-random（同预算随机顺序） | `--algo cqs --cqs-order random` |
| CQS-anti（同预算逆序） | `--algo cqs --cqs-order anti` |
| CQS + 静态权重 n_i^β（与 UDR 同集中度的对照） | `--algo cqs --reweight static --static-beta β` |
| CQS + UDR-FS（偏离含 BatchNorm 统计量） | `--algo cqs --reweight udr-fs` |
| FedBSS-pace | `--algo fedbss` |
| FedBSS-mean（低损失组 = 损失不超过本客户端均值的窗口） | `--algo fedbss --bss-split mean` |
| CQS + FedNova（按本地步数归一化的聚合，代替 UDR） | `--algo cqs --nova` |
| CQS + FedProx（本地加近端项） | `--algo cqs --prox-mu μ` |
| 其余基线 | `central`、`fedavg`、`fedprox`、`fednova`、`fedlc` |

- CQS：每轮用广播模型（eval 模式）给本客户端全部训练窗口打分并升序排序，一轮内各 epoch 共用这个排序。前 `--cqs-warmup`（默认 5）轮每个 epoch 用全部窗口；之后第 e 个本地 epoch 只训练损失最低的 ⌈n_i q_e⌉ 个窗口（epoch 内打乱），q_e 从 `--cqs-floor`（0.6）线性升到 1。E = 3 时配额为 0.6、0.8、1.0：损失最低的 60%、其后 20%、最高 20% 的窗口每轮分别访问 3、2、1 次，预热后每轮约 2.4 n_i 次访问（FedAvg 为 3 n_i）。
- 与原稿的一处口径差别：打分用的是训练损失本身，即物理位移平方误差（m²）；原稿用逐窗口归一化空间的损失。这正好对应原稿局限第四条提到的“按物理误差排序”，论文中要写明。
- CQS-random 每个客户端每轮重新抽一个随机顺序，访问次数分布与 CQS 完全相同。它不做打分前向，所以 timing 阶段两者的轮时差就是打分开销。
- UDR：只用可训练参数的更新，d_i = ‖Δ_i − 本轮参与客户端的等权平均更新‖，r_i = exp(−d_i / (median d · τ_d))，s_i = clip(r_i / Σ_j p_j r_j, 0.5, 1.5)，α_i ∝ n_i s_i；参数和浮点 buffer（BatchNorm 统计量）都按 α 聚合，p_i 按本轮参与者计算。每个客户端的权重在其样本份额的 1/3 到 3 倍之间，偏离全相等时与 FedAvg 完全相同。
- 记录：`log.csv` 的 `extra` 列新增 `udr_dev`、`udr_s`、`alpha`（UDR），以及所有 FedAvg 类方法的 `keff`（有效客户端数 1/Σα²）。

### 0.2 本地核对（2026-10-22）

- 原有 11 个配置（含部分参与）在 CPU 上与修改前的代码逐位相同：验证曲线、测试误差、分层误差和逐轮诊断都一致。
- CQS 配额与原稿代码 `get_fedbss_local_select_ratio` 一致（linear 和 cosine，E = 1–7）。UDR 与原稿 `compute_update_drift_weights`（可训练参数版）在 200 组随机输入上一致。权重界、偏离相等时退化为 FedAvg、对更新整体缩放不变，也都检查过。
- 真实数据：NGSIM 全量（GPU）预热后一轮访问 987,965 次，等于按各客户端实际窗口数算出的 Σ_i Σ_e ⌈n_i q_e⌉（2.4 × 411,650 = 987,960，取整多 5 次）。highD 60 段录像、10% 参与时每轮 6 个客户端，α/p 都在界内。
- 预热轮中 CQS、CQS-random 与 FedAvg 逐位相同，Fed-SSBR 与单独的 UDR 逐位相同；smoke 阶段的 11 个新任务在两套数据上都能跑通，逐轮访问量和权重都符合上面的定义。
- `--ssbr-select` 用伪造的 pilot 结果测过继续、止损和恰好差 0.02 m 的边界情况。
- 静态权重、UDR-FS、FedBSS-mean（同日晚加入）：原有 14 个配置（含 CQS、UDR、部分参与）在 CPU 上与加入前逐位相同。UDR-FS 与原稿 `compute_update_drift_weights`（全状态版）一致；BatchNorm 统计量不变时与 UDR 相同。静态权重 β = 1 时与 FedAvg 相同，参数和浮点 buffer 都按 n_i^β 聚合。FedBSS-mean 每轮的低损失组比例等于损失不超过本客户端均值的窗口占比。`--ssbr-select` 反解出的 β 使静态权重的 K_eff 与 Fed-SSBR 的平均 K_eff 相差不到 0.002。
- `--nova`、`--prox-mu`（10-24 加入）：原有 18 个配置在 CPU 上与加入前逐位相同。CQS 预热轮中，`--algo cqs --nova` 与 `--algo fednova`、`--algo cqs --prox-mu 0.01` 与 `--algo fedprox --mu 0.01` 逐位相同；预热后它们与 CQS 用同样的访问次数训练。`--nova` 的聚合与 FedNova 的公式及 `--algo fednova` 的聚合器逐位相同；在 highD 录像中步数不等的客户端上（11 步对 15 步），它与 FedAvg 聚合结果不同。

### 0.3 运行顺序

第 2、3 节（上传、环境检查）不变。之后：

| 顺序 | 阶段 | 任务数 | 说明 |
|---|---|---|---|
| 1 | smoke | 38 | 已加入 CQS、UDR 等新配置（不预热，1 轮小样本） |
| 2 | timing | 8 | 已加入 CQS、CQS-random、Fed-SSBR（不预热，3 轮）；服务器上不要同时跑别的任务 |
| 3 | ssbr-pilot | 28 | 种子 1；两个主设置：NGSIM 按时段（natural，6 个客户端）、highD 按运动模式（motion，10 个客户端）；10 个配置，FedProx 跑 μ = 1e-4、1e-3、1e-2、1e-1、1。后两个是第一次 pilot 之后补的：当时网格上界 1e-2 在两个设置上都最好（`run_pilot_ext.sh`） |
| 4 | — | — | `python -m fedtraj.jobs --ssbr-select`：打印继续/止损判断，按验证集选 FedProx 的 μ，写入 `runs_fedtraj/tuned_ssbr.json`。**先把输出发回来，和导师确认后再继续** |
| 5 | ssbr-main | 110 | 10 个配置的种子 2–5（种子 1 已在 pilot 中完成，自动跳过），之后 5 个关键配置（FedAvg、CQS、Fed-SSBR、CQS-random、选出 μ 的 FedProx）的种子 6–8 |
| 6 | ssbr-robust | 30 | 种子 1–3。NGSIM 按运动模式（原稿的划分方式）：FedAvg、CQS、CQS-random、UDR、Fed-SSBR；highD 60 段录像各为一个客户端、每轮 10% 参与、100 轮：FedAvg、FedNova、CQS、UDR、Fed-SSBR |
| 7 | ssbr-extra | 86 | 原稿其余对照，两个主设置。种子 1–5：CQS+静态权重、FedBSS-mean、CQS+UDR-FS；种子 1–3：CQS-anti，以及 FedAvg、CQS、UDR、Fed-SSBR 的半数参与；种子 1：Fed-SSBR 敏感性（q_min 0.4/0.8、T_w 0/10、τ_d 0.5/2、乘子界 [0.7,1.3]/[0.3,1.7]）加两次同种子重复；NGSIM 种子 1–3：E=5 的线性/余弦配额 |

| 8 | ssbr-combo | 20 | 10-24 看到主实验结果之后加入：CQS + FedNova、CQS + FedProx（μ 用 pilot 选出的值，不重新调），两个主设置，种子 1–5。由 `run_combo.sh` 在 `run_rest.sh` 结束后自动运行 |

- 静态权重的 β 由 `--ssbr-select` 从 pilot 中 Fed-SSBR（种子 1）全部 30 轮的平均 K_eff 反解，写入 `tuned_ssbr.json` 的 `static_beta`；ssbr-main 和 ssbr-extra 都读这个文件。
- pilot 通过、确认继续后，用 `run_rest.sh` 依次跑 ssbr-main → ssbr-robust → ssbr-extra（单卡 4 并发，日志追加到 `run_all.log`），最后生成 `results_vs_fedavg.txt`、`results_vs_cqs.txt`：

```bash
setsid nohup ./run_rest.sh >> run_all.log 2>&1 < /dev/null &
```

- 本地查看进度：双击仓库根目录的 `查看训练进度.bat`（只读）。

命令格式与第 4 节相同。单卡时用 `--mem-gb` 按显存装满 GPU（每个任务的显存按其数据集 `frames.npy` 加 `--overhead-gb` 估计，也可以用 `--job-gb` 给实测值），例如：

```bash
nohup python -m fedtraj.launch --stage ssbr-pilot --mem-gb 22 > launch_ssbr_pilot.log 2>&1 &
```

timing 阶段必须独占 GPU（`--per-gpu 1`，不要用 `--mem-gb`），否则轮时不能用于成本比较。

- highD 不用地点划分（natural）作主设置：一个地点占 82% 的训练窗口，联邦设定的意义很弱。
- 止损判据（跑之前已约定）：种子 1 上，如果 CQS 在两个主设置上的验证集 5 s 误差（最后 10 轮均值）都没有比 FedAvg 低 0.02 m 以上，就停止补种子，原稿按现有证据投 T-ITS。只看验证集，测试集留到最后报告。
- 汇总：`python -m fedtraj.collect --runs runs_fedtraj --ref fedavg` 给出相对 FedAvg 的配对差；UDR 在 CQS 之上的增量用 `--ref cqs` 再跑一次，看 `cqs-udr` 一行。
- 耗时（2026-10-22 在 RTX 4090 上实测）：timing 阶段独占 GPU 时 FedAvg 每轮 46.2 s、CQS 39.2 s，一个 30 轮任务约 23 分钟。单卡并发 2 个以上时总吞吐约为单任务的 1.7 倍，不再随进程数增加；4 并发时每个 NGSIM 任务约 1 小时、highD 约慢 30%。pilot 24 个实际用了 6.5 小时（平均每个任务约 16 分钟的 GPU 时间）；main、robust、extra 共 226 个约 58 小时（粗估，以实际为准）。

### 0.4 预先确定的分析（2026-10-22 23:40，ssbr-main 开始前、看到任何多种子结果之前写定）

- 主指标：每个运行在验证集 5 s RMSE 最低的那一轮选检查点，报告该检查点的测试 5 s RMSE。比较在同一设置内按种子配对，差值为“前者减后者”，给均值、样本标准差、双侧 95% t 区间和前者更低的种子数。区间不含零且各种子方向一致，才写成有效应；区间含零写成“分辨不出”。不做多重比较校正，论文里说明。
- 主要比较（两个主设置分别报告）：Fed-SSBR − FedAvg、CQS − FedAvg、CQS − CQS-random（顺序是否关键）、Fed-SSBR − CQS（UDR 的增量）、Fed-SSBR − FedProx（选出的 μ）。这 5 个配置跑种子 1–8；其余配置种子 1–5，用于次要比较。
- 次要指标：1–5 s 平均 RMSE、各时域 RMSE、最差客户端 5 s RMSE、CV 难度最高五分位（Q5）和换道窗口的 5 s RMSE。最后 10 轮的验证误差均值只用来描述训练后期的过拟合，不作为方法优劣的依据。pilot 判据用了这个量，是因为它对单种子的噪声不敏感；pilot 也显示各方法在第 7–24 轮达到最低验证误差，之后上升。
- 种子数：主要比较 8 个、其余 5 个，在看到 main 结果之前定下，之后不再按结果增减。pilot 的种子 1 计入，选 μ 和 β 时只用了它的验证集。
- 不按结果改协议：轮数（30）、本地 epoch（3）、学习率和 CQS/UDR 的超参数都保持 pilot 的设定。FedProx 的 μ 和静态权重的 β 只按 pilot 种子 1 的验证集选择一次。
- 主实验之后新增的对照（2026-10-24 写定，在这些运行开始之前）：主实验显示单独的 CQS 在 highD 上不如 FedProx 和 FedNova，所以补 CQS + FedNova、CQS + FedProx（ssbr-combo），检验 UDR 是否只是换一种已有的服务器或本地规则就能得到的效果。这组运行在论文中注明是“看到主实验结果之后加入的”，不论结果如何都报告。种子 1–5，之后不增减；μ 取 pilot 为 FedProx 选出的值（两个设置都是 1e-2），不为组合重新调参，论文里说明这一点可能略微不利于组合。比较：CQS+FedNova − Fed-SSBR、CQS+FedProx − Fed-SSBR（UDR 与已有规则相比），CQS+FedNova − FedNova、CQS+FedProx − FedProx（CQS 能否叠加在已有规则上），口径与上面相同。

### 0.5 R16 之后的补充检查（2026-10-26）

R16 的全部结果出来之后，针对审稿意见加的两组检查：同样访问次数下进入时机的作用，以及换一个预测骨干。阶段定义（`jobs.py` 的 `r17-*`）在这些运行开始之前写定，不论结果如何都报告，论文中注明是看到 R16 结果之后加入的。

| 论文中的名称 | 参数 |
|---|---|
| CQS 降序配额（同一排序，配额 1.0、0.8、0.6） | `--algo cqs --cqs-direction descending` |
| BatchNorm 统计量重估（本地 epoch 结束后用本客户端全部窗口） | `--bn-recal all` |
| LSTM 骨干（无 CV 锚） | `--backbone lstm --eval-batch 1024` |

- 降序配额：排序与 CQS 相同，每轮算一次；各 epoch 的窗口集合是 CQS 的倒序，所以每个窗口的访问次数与 CQS 完全相同（3 : 2 : 1），只是损失最高的 20% 在第一个 epoch 使用，最后一个 epoch 只有损失最低的 60%。
- BN 重估：PyTorch 默认 momentum 0.1，上传的 running 统计量几乎都来自最后一个 epoch，降序时它只含低损失窗口。重估在本地 epoch 之后，用本客户端全部窗口做一遍训练模式的前向（不求梯度，累积平均），然后恢复 momentum。主比较的升序、降序都重估；两者再各跑 3 个不重估的种子，看重估本身的影响。
- LSTM：Deo 和 Trivedi（2018）V-LSTM 的结构（输入嵌入、单层 LSTM 编码器、编码隐状态重复作为每步解码输入、单层 LSTM 解码器、线性输出），输入同样的 44 通道帧序列，输出按训练集位移的逐步均值和标准差标准化的位移，不加 CV 外推；默认初始化，没有 BatchNorm 和 dropout，210,050 个参数。评估和打分的 batch 为 1024：cuDNN LSTM 在 4096 时需要约 2.4 GB 额外显存，batch 大小不改变逐窗口预测。
- 学习率：只按 FedAvg 的验证误差（种子 1）在 5e-4、1e-3、2e-3 中选，`--r17-select` 写入 `runs_r17/tuned_r17.json`；选出的 FedAvg 在验证集上不低于 CV 时不跑 LSTM 主实验。结果：NGSIM 选 5e-4（网格下端），highD 选 1e-3，两者都低于 CV。

本地核对（10-26）：

- 降序与升序：同一客户端上每个窗口的访问次数相同，各 epoch 的窗口集合正好是升序的倒序；损失最高的 20% 只访问一次，升序时在最后一个 epoch，降序时在第一个。
- BN 重估：6 个 BatchNorm 层都重估，累积的 batch 数等于 ⌈n_i / 512⌉，第一层的 running mean 与逐 batch 均值的平均相差 4.5e-8，之后 momentum 恢复为 0.1。
- 原有 18 个配置在 CPU 上与修改前逐位相同（测试误差、分层误差、验证曲线和逐轮日志）；此前 12 个阶段的任务列表（共 630 个）不变。
- r17-smoke（16 个 2 轮小样本任务）在两套数据上都能跑通。

运行顺序（第二台服务器，两块 RTX 4090，见 `tools/server/README.md`，由 `tools/server/run_r17.sh` 依次执行）：

| 顺序 | 阶段 | 任务数 | 说明 |
|---|---|---|---|
| 1 | r17-smoke | 16 | 小样本 2 轮，任何一个失败就停止（输出在 `runs_r17_smoke/`） |
| 2 | r17-lstm-pilot | 6 | LSTM 的 FedAvg，种子 1，三个学习率 |
| 3 | — | — | `python -m fedtraj.jobs --r17-select` |
| 4 | r17-main | 82 | r17-order 44 个（种子 1–8：CQS+BN、CQS 降序+BN；种子 1–3：CQS 降序、CQS），然后 r17-lstm 38 个（FedAvg、CQS、FedBSS-fixed、CQS-random，种子 1–5；FedAvg 的种子 1 已在 pilot 中完成） |

结果（88 个运行）在仓库根目录的 `data/fedtraj_results_r17_20261026_runs.tar.gz` 和 `_ckpt.tar.gz`，解包到 `results_r17/`；配对比较用 `paper/scripts/analyze_r17.py`，口径与第 0.4 节相同。

## 1. 方法与配置对照

FedTR（备用方案，留作下一篇；本次论文见第 0 节）的统一原理是"目标不变的调度"：两层都只改变优化路径，SGD 一阶意义下不改变联邦目标。

- 样本层（尾部路由）：每轮用广播模型 θ^t 给本地窗口打分（物理位移误差，m²），损失最高的 q_i 为尾部。第 1 个本地 epoch 训练全部窗口，尾部损失权重为 E = 3；第 2…E 个 epoch 只训练其余窗口。每个窗口每轮的总权重与 FedAvg 相同，只是尾部集中在离 θ^t 最近的位置训练。
- 客户端层（漂移最优预算）：因为有权重补偿，q_i 取任何值都不改变一阶目标，服务器可以按客户端分配。每个窗口每轮平均访问 s_i = 1 + (E−1)(1 − q_i)；在总预算 Σ p_i s_i = 2.6 下最小化局部 SGD 的二阶漂移项 Σ p_i s_i² ζ_i²，闭式解是 s_i ∝ 1/ζ_i²。ζ_i² 用 d_i = 1 − EMA(cos) 估计，cos 是该客户端更新与其余客户端平均更新的余弦（留一法，只用已上传的模型）。q_i 裁剪到 [0.05, 0.6]，裁剪后用二分法保证预算约束精确成立。漂移大的客户端少重复本地窗口，漂移小的多走几步。平均每轮训练访问 2.6 n_i（FedAvg 3 n_i，FedBSS 2.6 n_i）。
- 一致性聚合（组件，非本文贡献）：上传累积对角预条件 N_i = lr·Σ_k 1/(√v̂_k + ε)，服务器按 p_i·N_s/N_i 逐坐标加权，即 Wang et al. (2021) 的 local + global correction。它消除 AdamW 下本地工作量不同带来的残余不一致。

| 论文中的名称 | 参数 |
|---|---|
| FedTR（两层） | `--algo fedtr` |
| 去掉客户端层预算 | `--algo fedtr --budget fixed` |
| 仅样本层 | `--algo fedtr --budget fixed --agg avg` |
| 仅借用的聚合规则 | `--algo fedlc` |
| 消融 | `--budget inverse`、`--agg avg`、`--route random`、`--tail-pos late`、`--tail-weight one`、`--precond tensor`、`--tail-frac 0.1/0.3` |
| 基线 | `central`、`fedavg`、`fedprox`、`fednova`、`fedavgm`、`fedadam`、`scaffold`、`fedbss` |

每轮的 q_i 记录在 `log.csv` 的 `extra` 列（`q`、`cos_loo`），可以直接画预算随轮次的变化。

所有方法共用：CV 锚定的同一骨干（0.84 M 参数）、物理空间损失、AdamW（lr 5e-4、wd 1e-4）、batch 512、30 轮 × 3 个本地 epoch（集中式 90 个 epoch），按验证集 5 s RMSE 选检查点。

## 2. 上传

本地执行 `python -m fedtraj.pack`，在 `upload_fedtraj/` 生成两个文件：

- `fedtraj_code.tar.gz`：代码，约 40 KB
- `fedtraj_data.tar`：`data/fedtraj_ngsim`（1.46 GB）和 `data/fedtraj_highd`（2.68 GB）

在服务器上新建项目目录，放入这两个文件后解包：

```bash
mkdir -p ~/fedtraj_proj && cd ~/fedtraj_proj
tar xzf fedtraj_code.tar.gz
tar xf fedtraj_data.tar
```

之后所有命令都在这个目录下执行（代码按相对路径读取 `data/...`）。只改了代码时，本地运行 `python -m fedtraj.pack --code-only`，重新上传并解包代码包即可。

## 3. 环境

只需要 Python ≥ 3.8、带 CUDA 的 PyTorch ≥ 2.0 和 numpy（pandas、scikit-learn 只在本地预处理时用）。本地测试版本：Python 3.8.20、torch 2.4.1+cu118、numpy 1.24.4。

```bash
python -m fedtraj.server_check
```

它会检查 GPU、逐文件核对 SHA-256，并实测一步训练的耗时，估算单个 30 轮任务的时间。输出 `ready` 后再继续。

## 4. 运行顺序

每个阶段一条命令。用 nohup 或 tmux，断开 SSH 后任务继续：

```bash
nohup python -m fedtraj.launch --stage smoke > launch_smoke.log 2>&1 &
tail -f launch_smoke.log
```

下表是 FedTR 的阶段；Fed-SSBR 的阶段见第 0.3 节，smoke 和 timing 两者共用。

| 顺序 | 阶段 | 任务数 | 说明 |
|---|---|---|---|
| 1 | smoke | 38 | 每个配置 1 轮小样本，几分钟，检查能否跑通（含 Fed-SSBR 的配置） |
| 2 | timing | 8 | 3 轮完整训练，用于成本对比；服务器上不要同时跑别的任务 |
| 3 | pilot | 24 | 种子 1，三个设置。**跑完先把结果发给我，再决定是否继续** |
| 4 | tune | 24 | 基线超参数；跑完执行 `python -m fedtraj.jobs --select-tuned` |
| 5 | main | 约 84 | 5 个种子（与 pilot、tune 重复的任务自动跳过） |
| 6 | ablation | 约 52 | |
| 7 | robust | 约 79 | 运动划分、半数参与、highD 60 段录像各为一个客户端；每个设置都有固定预算对照 |

- launcher 自动使用所有可见 GPU。`--per-gpu` 控制每张卡同时跑几个任务，默认 2（timing 为 1），24 GB 显存可设 3。
- 已完成的任务（有 `result.json`）会自动跳过。中断后重新执行同一条命令即可续跑，未完成的任务从第 1 轮重跑。失败的任务自动重试一次，日志在 `logs_fedtraj/<stage>/`。
- 预览将要运行的任务：`python -m fedtraj.launch --stage main --dry-run`
- 停止全部任务：`pkill -f fedtraj.launch; pkill -f fedtraj.train`

## 5. pilot 之后的判断（FedTR；Fed-SSBR 用第 0.3 节的 `--ssbr-select`）

```bash
python -m fedtraj.collect --runs runs_fedtraj --ref fedavg
```

判断只看验证集两列（`val_best`、`val_last10`），测试集留到最后报告：

- central 与 fedavg 的差距：如果 highD-natural 上几乎为零（该划分中一个地点占 82% 的训练窗口），highD 的主实验改用 motion 或 recording 划分。
- fedtr 对 fedavg：至少在 3 个设置中的 2 个上 `val_last10` 更低，才进入 tune 和 main。
- 样本层是否有贡献：fedtr-bfixed-aggavg 对 fedavg、fedtr-bfixed 对 fedlc。
- 客户端层是否有贡献：fedtr 对 fedtr-bfixed，同时看 `max_client_gap`（客户端相对集中式的最大超出比例）是否缩小。
- 预热：fedtr 对 fedtr-tw5（前 5 轮不路由）。本地小规模试跑显示从 CV 初始化直接路由会拖慢前几轮，哪个更好由验证集决定，之后的阶段统一用胜出的版本（由我改代码默认值后重新打包）。
- 单个种子差距小于约 0.02 m 时视为分辨不出，留给 5 个种子的 main 和 ablation 判断。

不满足时先停下，把结果发给我，免得在 main 阶段白花约 150 GPU 小时。

## 6. 回传结果

```bash
tar czf results.tgz --exclude='*.pt' runs_fedtraj runs_timing logs_fedtraj
```

`best.pt` 是检查点，需要时再单独传。每个任务目录里有 `log.csv`（逐轮验证误差、访问次数、客户端更新范数和与聚合方向的余弦）和 `result.json`（测试误差、逐客户端误差、按 CV 难度五分位和换道窗口的分层误差）。轮时汇总：`python -m fedtraj.collect --runs runs_timing --timing`。

## 7. 耗时与显存

- 显存：每个任务 NGSIM 约 2.6 GB、highD 约 3.5 GB（整份数据放在显存里）。
- 本地 RTX 3050 笔记本上，NGSIM 一轮约 210 s。4090 预计快 3–5 倍，单个 30 轮任务约 30–50 分钟，全部约 260 个任务合计约 170–220 GPU 小时（未实测）。以 `server_check` 和 timing 阶段的实测为准。
- CQS 每轮多一次打分前向：本地 RTX 3050 上给全部 411,650 个 NGSIM 训练窗口打分约 19 s（同一张卡上训练一步约 87 ms，FedAvg 一轮的训练部分约 211 s）。

## 8. 数据（已处理好，服务器上不需要重做）

- NGSIM：读取 6 个 15 分钟时段的完整 txt（zip 里的 csv 在 1,048,575 行处被截断），共 11,779 辆车。按车辆 8:1:1 划分（按时段分层），窗口步长 15 帧；训练 / 验证 / 测试 411,650 / 51,491 / 50,121 个窗口。
- highD：60 段录像，25 Hz 线性插值到 10 Hz，转到行驶方向坐标系。按车辆 8:1:1 划分（按录像分层），步长 10 帧；525,875 / 66,135 / 66,222 个窗口。
- 每帧 44 个通道：自车位置、速度、加速度，6 个邻车槽的相对位置、速度、加速度，左右车道是否存在。速度和加速度由位置做 0.5 s 因果后向差分得到，两个数据集口径一致。
- 划分：natural（NGSIM 按时段 6 个、highD 按地点 6 个），motion（按观测段运动特征 K-means 10 个），recording（highD 60 段录像各为一个客户端）。
- 本地重建：`python -m fedtraj.prep_ngsim`、`python -m fedtraj.prep_highd`；之后运行 `python -m fedtraj.manifest write <数据目录>` 更新校验文件，`python -m fedtraj.stats_report --data-dir <数据目录>` 输出与模型无关的统计。
