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
