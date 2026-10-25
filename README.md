# Code and data: Curriculum-Quota Scheduling for Federated Vehicle Trajectory Prediction

Training code, the processed NGSIM dataset, all 274 experiment runs, and the scripts and statistics behind every
number, table, and data figure of the manuscript (revision R12, 25 October 2026).

| Path | Content | Size |
|---|---|---|
| `data/fedtraj_results_20261025_part1.tar.gz` (Git LFS) | `result.json` and `log.csv` of all 274 runs and of the 8 timing runs, launcher and training logs, summary tables, and 80 of the 274 checkpoints (`best.pt`) | 243 MiB |
| `data/fedtraj_results_20261025_part2.tar.gz`, `part3` (Git LFS) | the other 194 checkpoints | 309 + 268 MiB |
| `data/fedtraj_ngsim.tar.gz.part01`-`part04` (Git LFS) | processed NGSIM dataset (`frames.npy`, window and track indices, client partitions, `meta.json`, `MANIFEST.json`), one tar.gz split into four parts | 902 MiB |
| `code/fedtraj/` | preprocessing, training, job lists, launcher, and summaries; section 0 of `code/fedtraj/README.md` lists the configurations, the run order, and the analysis fixed before the multi-seed runs | |
| `code/tools/dataset_parts.py` | checks and unpacks the dataset parts (and shows how they were made) | |
| `code/tools/server/` | shell scripts used on the training server, with the run log | |
| `paper/scripts/` | statistics, tables, data figures, trajectory examples, and the number and style checks of the manuscript | |
| `paper/data/` | `stats_r12.json` (every reported statistic), `figdata_r12.json` (figure data), `difficulty_r12.json` (model-free client statistics), and a readable dump | |
| `paper/data/qualitative/` | trajectory examples: selection rule, per-window errors of all Q5 and lane-change test windows, and the trajectories and raw windows of the selected NGSIM examples | |
| `paper/figures/fig_traj.*`, `paper/fig_traj_snippet.tex` | trajectory figure, with its LaTeX figure environment and paragraph | |
| `paper/dataset_meta/` | `meta.json`, `MANIFEST.json` (size and SHA-256 of every file), and model-free statistics of the two processed datasets | |
| `paper/fonts_figures/` | URW Nimbus Roman fonts used in the figures, with their licenses | |
| `SHA256SUMS.txt` | SHA-256 of every Git LFS file | |

## Download

The Git LFS bandwidth of this repository is limited, so clone without the large files and fetch only what you need:

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ShamG1/Fed_Paper_data.git     # PowerShell: $env:GIT_LFS_SKIP_SMUDGE=1; git clone ...
cd Fed_Paper_data
git lfs pull --include "data/fedtraj_results_*"           # all runs, with checkpoints (820 MiB)
git lfs pull --include "data/fedtraj_ngsim.tar.gz.part*"  # processed NGSIM dataset (902 MiB)
sha256sum -c --ignore-missing SHA256SUMS.txt              # Linux / macOS
```

## Reproduce the statistics

Requirements: Python 3.8 or later with numpy and scipy (matplotlib for the figures; torch 2.0 or later for
training and for the trajectory examples).

```
tar -xzf data/fedtraj_results_20261025_part1.tar.gz
tar -xzf data/fedtraj_results_20261025_part2.tar.gz
tar -xzf data/fedtraj_results_20261025_part3.tar.gz
python paper/scripts/make_stats_r12.py --results results_fedtraj_20261025 --data paper/dataset_meta
python paper/scripts/make_tables_r12.py
```

The three archives unpack into one directory, `results_fedtraj_20261025/`. `make_stats_r12.py` rewrites
`paper/data/stats_r12.json` and `figdata_r12.json`; the rewritten files are identical to the committed ones, so
`git status` shows no change.

## Processed NGSIM dataset and training

```
python code/tools/dataset_parts.py extract fedtraj_ngsim    # checks the parts, unpacks to code/data/fedtraj_ngsim/
cd code
python -m fedtraj.train --data-dir data/fedtraj_ngsim --partition natural --algo cqs --seed 1
```

`extract` checks every part against `SHA256SUMS.txt` and every unpacked file against `MANIFEST.json`. On Linux
and macOS, `cat data/fedtraj_ngsim.tar.gz.part* | tar -xzf - -C code` unpacks the same files. Commands are run
from `code/`, where `data/fedtraj_ngsim` is the default dataset path of `fedtraj.jobs`. The run ids and the
options of every experiment are in `code/fedtraj/README.md` and `code/fedtraj/jobs.py`.

The processed highD dataset is not in this repository. Rebuild it from the raw highD data, from `code/`:

```
python -m fedtraj.prep_highd --src <highD>/data --out data/fedtraj_highd
python -m fedtraj.manifest write data/fedtraj_highd
```

`paper/dataset_meta/fedtraj_highd/MANIFEST.json` holds the size and SHA-256 of every file of the highD dataset used
in the paper, for comparison with a rebuild (`meta.json` also records its creation time, so it always differs). The NGSIM dataset is rebuilt in the same way with
`python -m fedtraj.prep_ngsim --src <folder with the US-101 and I-80 zip files> --out data/fedtraj_ngsim`.

## Trajectory examples

`paper/data/qualitative/SELECTION_RULE.md` fixes which test windows are shown, before any per-window prediction
was computed: in each main setting (NGSIM-natural, highD-motion) and each stratum (Q5, the quintile with the
largest constant-velocity error at 5 s; LC, windows with more than 3 m of lateral displacement at 5 s), the windows
nearest to the 10th, 50th, and 90th percentiles of the 5 s error of CQS minus that of FedAvg, for the seed-one
checkpoints. `paper/scripts/make_qualitative_r12.py` applies the rule. It needs torch, the checkpoints, and the
processed datasets, and it first checks that its per-window errors reproduce the test, Q5, and lane-change RMSE in
each `result.json` (largest difference 3.4e-4 m). `qualitative_r12.json` records the checks, the stratum sizes, and
the selected windows.

## Data sources and licenses

- NGSIM US-101 and I-80: U.S. Department of Transportation, Federal Highway Administration (public data). The
  processed NGSIM dataset in `data/` is derived from it.
- highD: free for non-commercial use, with access granted per request by levelXdata
  (https://levelxdata.com/highd-dataset/). The processed highD data and the raw highD windows of the trajectory
  examples are not included. The highD checkpoints, the highD per-window errors, and the six highD windows drawn in
  the trajectory figure are derived from highD.
