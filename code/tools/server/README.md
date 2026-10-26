# 服务器上的运行脚本（记录用）

服务器：RTX 4090（24 GB）单卡，项目目录 `/data/fedtraj_proj`，环境 `~/anaconda3/envs/torch2.4_cuda12.1`（Python 3.10、torch 2.4.1+cu121、numpy 2.0.1）。

| 文件 | 作用 | 状态 |
|---|---|---|
| `run_all.sh` | smoke → timing（独占 GPU）→ ssbr-pilot（4 并发）→ `--ssbr-select` 与汇总表，然后停下 | 10-22 16:08 启动，23:03 完成（24/24，无失败） |
| `run_pilot_ext.sh` | FedProx 网格补 μ = 1e-1、1（种子 1，两个设置，4 个任务），然后重新做 `--ssbr-select` 和汇总表 | 10-22 23:17 启动，10-23 00:32 完成。μ = 1e-1、1 在两个设置上都更差，仍选 μ = 1e-2（网格内部的最优值） |
| `run_rest.sh` | ssbr-main（110）→ ssbr-robust（30）→ ssbr-extra（86），最后生成 `results_vs_fedavg.txt`、`results_vs_cqs.txt`。先等 `run_pilot_ext.sh` 结束；判断为 STOP、或 `ssbr_select.txt` 没有覆盖扩展后的 μ 网格时不启动 | 10-22 23:36 启动（确认继续、加种子之后）。ssbr-main 10-23 00:32 → 10-24 05:37（110/110，无失败）；ssbr-robust 05:37 → 10:36（30/30，无失败）；ssbr-extra 10-24 10:36 → 10-25 06:11（86/86，无失败） |
| `run_combo.sh` | 主实验之后加入的 ssbr-combo（CQS + FedNova、CQS + FedProx，20 个任务），先等 `run_rest.sh` 结束，最后重写两张汇总表 | 10-24 11:01 启动，10-25 06:11 → 11:41（20/20，无失败） |

全部 274 个运行在 10-25 11:41 完成，没有失败、没有发散。本地备份（10-25，MD5 已与服务器核对）：

- `results_fedtraj_20261025.tgz`（仓库根目录，5.4 MB）：全部 `result.json`、`log.csv`、训练日志、各阶段 launcher 日志、`ssbr_select*.txt`、汇总表和 `analysis_final.txt`（按 `fedtraj/README.md` 第 0.4 节口径的全部比较），不含检查点。
- `results_fedtraj_20261025/`：上面的包解压后的目录；274 个 `best.pt` 检查点（889 MB）也已下载并放进对应的运行目录。

手动启动 `run_rest.sh`（在服务器项目目录下；已完成的任务会自动跳过，中断后可重新执行）：

```bash
setsid nohup ./run_rest.sh >> run_all.log 2>&1 < /dev/null &
```

本机查看进度：双击仓库根目录的 `查看训练进度.bat`（只读）。分析口径见 `fedtraj/README.md` 第 0.4 节（在 main 开始前写定）。

代码替换记录（替换前都在服务器上编译并做过检查，旧代码有备份）：

- 10-22 18:55，pilot 运行中：加入静态权重、UDR-FS、FedBSS-mean 和 ssbr-extra 阶段。替换前在 CPU 上确认原有 14 个配置的结果逐位不变。备份 `fedtraj_code_backup_1022_1855.tar.gz`。
- 10-22 23:17，pilot 结束后：FedProx 网格扩展到 μ = 1e-4 … 1（只改 `jobs.py`）。备份 `fedtraj_code_backup_1022_2317.tar.gz`；第一次 pilot 的判断结果保存为 `ssbr_select_pilot1.txt`。
- 10-22 23:36：ssbr-main 加入 5 个关键配置的种子 6–8，README 写入预先确定的分析（只改 `jobs.py`、`README.md`）。备份 `fedtraj_code_backup_1022_2336.tar.gz`。
- 10-24 11:01，ssbr-extra 运行中：加入 `--nova`、`--prox-mu` 和 ssbr-combo 阶段（改 `train.py`、`jobs.py`、`README.md`）。替换前在 CPU 上确认原有 18 个配置的结果逐位不变，并在服务器上确认 pilot、main、robust、extra 的任务列表不变。备份 `fedtraj_code_backup_1024_1101.tar.gz`。

## 第二台服务器（2026-10-26，R16 之后的补充检查）

两块 RTX 4090（24 GB），项目目录 `/root/fedtraj_proj/code`，环境 `/root/envs/t24`（conda 建的 Python 3.10.22，pip 装 torch 2.4.1+cu121、numpy 2.0.1，torch 版本与第一台相同）。数据用仓库里的 Git LFS 分卷上传，`sha256sum -c` 和 `python -m fedtraj.manifest verify` 都通过。服务器时钟是 UTC，下表是北京时间。

| 文件 | 作用 | 状态 |
|---|---|---|
| `run_r17.sh` | r17-smoke（任何一个失败就停）→ r17-lstm-pilot → `--r17-select` → r17-main（r17-order，然后 r17-lstm） | 10-26 11:06 启动；smoke 16/16，pilot 6/6（11:07–11:23），main 82/82（11:23–18:10），无失败、无发散 |

- 显存：每块卡按 22.5 GB 装任务，每卡最多 4 个。CV 锚定骨干用第一台的值（NGSIM 4.5 GB、highD 6.0 GB）；LSTM 用这台实测的峰值加 CUDA 上下文（3.7、4.9 GB，`--job-gb lstm:<数据集>=GB`）。第一次手动试跑 smoke 时，两个 highD-LSTM 任务在验证时显存不足，重试后通过；之后把 LSTM 的评估 batch 改为 1024，并改用上面的显存估计，`run_r17.sh` 里的 smoke 一次通过。
- 代码：`run_r17.sh` 启动前上传的代码与仓库中的提交相同（各文件的修改时间都早于上传）。
- 结果：服务器上打包的 `results_r17.tgz` 下载后与服务器核对过 SHA-256；按“结果与日志”“检查点”拆成仓库根目录的 `data/fedtraj_results_r17_20261026_runs.tar.gz` 和 `_ckpt.tar.gz`（都解包到 `results_r17/`），拆分后逐文件与原包一致。
