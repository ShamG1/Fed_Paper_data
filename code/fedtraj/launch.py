"""Run one experiment stage on all GPUs of the machine, several jobs per GPU, resumable.

Every job is ``python -m fedtraj.train ...`` with CUDA_VISIBLE_DEVICES set to one GPU; its output goes to
``<log-dir>/<stage>/<name>.log``.  A job counts as done when its result.json exists, so the launcher can be
stopped and restarted at any time (unfinished runs start again from round 1).  Failed jobs are retried once.

Usage (from the project root; keep it running after logout with nohup or tmux)
    nohup python -m fedtraj.launch --stage pilot --per-gpu 3 > launch_pilot.log 2>&1 &
    tail -f launch_pilot.log
    python -m fedtraj.launch --stage main --dry-run          # list what would run
Stop everything:  pkill -f fedtraj.launch; pkill -f fedtraj.train

Memory packing (``--mem-gb``): instead of a fixed number of jobs per GPU, start a job on a GPU while the
estimated memory of the jobs running there stays within the budget, up to ``--per-gpu`` jobs (default 8 in this
mode).  A job's estimate is the size of its dataset's frames.npy (the whole dataset lives on the GPU) plus
``--overhead-gb`` (activations, CUDA context), or a measured value from ``--job-gb name=GB,...``.  Queued jobs
are taken in order, but a smaller job may start ahead of one that does not fit.  ``--stagger`` spaces out the
starts on one GPU so that data loading does not pile up.
    nohup python -m fedtraj.launch --stage ssbr-pilot --mem-gb 23 > launch_ssbr_pilot.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtraj.jobs import STAGES, build_jobs  # noqa: E402


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        return [str(i) for i in range(n)] or ["0"]
    except (OSError, subprocess.CalledProcessError):
        return ["0"]


def stamp():
    return time.strftime("%m-%d %H:%M:%S")


def job_memory(job, table, overhead):
    """Estimated GPU memory (GB) of a job: measured value for its dataset, else frames.npy + overhead."""
    argv = job["argv"]
    data = Path(argv[argv.index("--data-dir") + 1])
    if data.name in table:
        return table[data.name]
    frames = data / "frames.npy"
    return (frames.stat().st_size / 2 ** 30 if frames.exists() else 0.0) + overhead


def parse_table(text):
    out = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        name, gb = item.split("=")
        out[name.strip()] = float(gb)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all)")
    ap.add_argument("--per-gpu", type=int, default=None,
                    help="concurrent jobs per GPU (default 2; timing 1); with --mem-gb the upper limit (default 8)")
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="pack jobs by estimated GPU memory within this budget per GPU (GB)")
    ap.add_argument("--overhead-gb", type=float, default=2.0,
                    help="--mem-gb: per-job memory on top of the dataset's frames.npy (GB)")
    ap.add_argument("--job-gb", default="",
                    help="--mem-gb: measured per-job memory by dataset directory name, e.g. fedtraj_ngsim=3.4")
    ap.add_argument("--stagger", type=float, default=None,
                    help="minimum seconds between two starts on one GPU (default 20 with --mem-gb, else 0)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ngsim", default="data/fedtraj_ngsim")
    ap.add_argument("--highd", default="data/fedtraj_highd")
    ap.add_argument("--log-dir", default="logs_fedtraj")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args.stage, args.out_dir, args.ngsim, args.highd)
    gpus = args.gpus.split(",") if args.gpus else visible_gpus()
    packing = args.mem_gb is not None
    per_gpu = args.per_gpu or (8 if packing else (1 if args.stage == "timing" else 2))
    stagger = args.stagger if args.stagger is not None else (20.0 if packing else 0.0)
    table = parse_table(args.job_gb)
    need = {j["name"]: (job_memory(j, table, args.overhead_gb) if packing else 0.0) for j in jobs}
    budget = (f", memory budget {args.mem_gb:g} GB per GPU, at most {per_gpu} jobs" if packing
              else f" x {per_gpu}")
    print(f"[{stamp()}] stage {args.stage}: {len(jobs)} runs to do on GPUs {gpus}{budget}", flush=True)
    if args.dry_run or not jobs:
        for j in jobs:
            print("   ", (f"[{need[j['name']]:.1f} GB] " if packing else "") + " ".join(j["argv"]))
        return
    log_dir = Path(args.log_dir) / args.stage
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [(j, 0) for j in jobs]
    running = {}                      # serial -> (Popen, job, attempt, start time, log handle, gpu)
    serial = 0
    done, failed = [], []
    while queue or running:
        for gpu in gpus:
            while queue:
                here = [r for r in running.values() if r[5] == gpu]
                if len(here) >= per_gpu:
                    break
                if stagger and here and time.time() - max(r[3] for r in here) < stagger:
                    break
                used = sum(need[r[1]["name"]] for r in here)
                # first queued job that fits the memory budget (a lone job always starts)
                pick = next((i for i, (j, _) in enumerate(queue)
                             if not packing or not here or used + need[j["name"]] <= args.mem_gb), None)
                if pick is None:
                    break
                job, attempt = queue.pop(pick)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                fh = open(log_dir / f"{job['name']}.log", "a", encoding="utf-8")
                fh.write(f"\n===== {stamp()} attempt {attempt + 1} on GPU {gpu}\n")
                fh.flush()
                p = subprocess.Popen([args.python, "-m", "fedtraj.train"] + job["argv"], stdout=fh,
                                     stderr=subprocess.STDOUT, env=env)
                running[serial] = (p, job, attempt, time.time(), fh, gpu)
                serial += 1
                mem = (f"; {len(here) + 1} running, {used + need[job['name']]:.1f}/{args.mem_gb:g} GB"
                       if packing else "")
                print(f"[{stamp()}] start  GPU{gpu}  {job['name']}  (queue {len(queue)}{mem})", flush=True)
        time.sleep(5)
        for s in list(running):
            p, job, attempt, t0, fh, gpu = running[s]
            if p.poll() is None:
                continue
            fh.close()
            del running[s]
            minutes = (time.time() - t0) / 60
            if (Path(job["run_dir"]) / "result.json").exists():
                done.append(job["name"])
                res = json.loads((Path(job["run_dir"]) / "result.json").read_text(encoding="utf-8"))
                print(f"[{stamp()}] done   {job['name']}  {minutes:.1f} min  test 5s {res['test']['5s']:.4f}  "
                      f"[{len(done)}/{len(jobs)}]", flush=True)
            elif attempt < args.retries:
                queue.append((job, attempt + 1))
                print(f"[{stamp()}] FAILED {job['name']} (exit {p.returncode}), retry later", flush=True)
            else:
                failed.append(job["name"])
                print(f"[{stamp()}] FAILED {job['name']} (exit {p.returncode}), giving up; "
                      f"see {log_dir / (job['name'] + '.log')}", flush=True)
    print(f"[{stamp()}] stage {args.stage} finished: {len(done)} done, {len(failed)} failed", flush=True)
    for name in failed:
        print("   failed:", name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
