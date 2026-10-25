"""Collect result.json files into tables with seed-paired differences, one table per (dataset, partition).

Configurations are run ids without the ``_s<seed>`` suffix.  Columns: test RMSE at 1-5 s and the 1-5 s mean,
worst-client 5 s RMSE, the largest relative excess of a client's 5 s RMSE over the centralized model
(``max_client_gap``, needs a ``central`` run in the same setting), 5 s RMSE of the hardest CV quintile (Q5) and of
lane-change windows, plus the best validation 5 s RMSE and the mean validation 5 s RMSE over the last 10 rounds
(less noisy, for pilot decisions).
With ``--ref``, paired differences (configuration minus reference on shared seeds) with two-sided 95% t
intervals and the number of seeds on which the configuration is lower.

Usage
    python -m fedtraj.collect --runs runs_fedtraj --ref fedavg [--csv results.csv]
    python -m fedtraj.collect --runs runs_timing --timing
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
KEYS = ["1s", "2s", "3s", "4s", "5s", "mean", "worst_client_5s", "max_client_gap", "Q5_5s", "lateral_5s",
        "val_best", "val_last10"]
DIFF_KEYS = ["5s", "mean", "worst_client_5s", "max_client_gap", "Q5_5s", "val_last10"]


def add_client_gaps(settings):
    """max_client_gap: largest relative 5 s RMSE excess of a client over the centralized model
    (same seed when available, otherwise the centralized seed mean); NaN without a centralized run."""
    for groups in settings.values():
        central = groups.get("central", {})
        mean_pc = (np.mean([r["_pc"] for r in central.values()], axis=0) if central else None)
        for name, runs in groups.items():
            for s, r in runs.items():
                ref = central[s]["_pc"] if s in central else mean_pc
                if ref is None:
                    r["max_client_gap"] = float("nan")
                    continue
                pc, ref = np.asarray(r["_pc"]), np.asarray(ref)
                ok = (pc > 0) & (ref > 0)
                r["max_client_gap"] = float(np.max(pc[ok] / ref[ok] - 1.0)) if ok.any() else float("nan")


def load(root):
    settings = {}
    for f in sorted(Path(root).glob("*/*/*/result.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        m = re.match(r"(.+)_s(\d+)$", r["run_id"])
        if not m:
            continue
        t = r["test"]
        curve = [c["5s"] for c in r["val_curve"]]
        row = {k: t[k] for k in ("1s", "2s", "3s", "4s", "5s", "mean")}
        row["worst_client_5s"] = max(v for v, n in zip(t["per_client_5s"], t["per_client_n"]) if n > 0)
        row["_pc"] = [v if n > 0 else 0.0 for v, n in zip(t["per_client_5s"], t["per_client_n"])]
        row["Q5_5s"] = r["strata"]["Q5"]
        row["lateral_5s"] = r["strata"]["lateral"]
        row["val_best"] = r["best_val_5s"]
        row["val_last10"] = float(np.mean(curve[-10:]))
        row["best_round"] = r["best_round"]
        setting = f"{f.parts[-4]}/{f.parts[-3]}"
        settings.setdefault(setting, {}).setdefault(m.group(1), {})[int(m.group(2))] = row
    add_client_gaps(settings)
    return settings


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():.3f}+-{v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"


def round_times(root):
    """Mean round time (s) over rounds 2..end and training visits per round, from log.csv of every run."""
    print(f"{'run':60s} {'rounds':>6s} {'s/round':>9s} {'visits/round':>13s}")
    for f in sorted(Path(root).glob("*/*/*/log.csv")):
        rows = list(csv.DictReader(line for line in open(f, encoding="utf-8") if not line.startswith("#")))
        if len(rows) < 2:
            continue
        t = [float(r["time_s"]) for r in rows[1:]]
        v = [float(r["visits"]) for r in rows[1:]]
        print(f"{'/'.join(f.parts[-4:-1]):60s} {len(rows):6d} {np.mean(t):9.1f} {np.mean(v):13.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_fedtraj")
    ap.add_argument("--ref", default="fedavg")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--timing", action="store_true", help="print round times from log.csv instead (runs_timing)")
    args = ap.parse_args()
    if args.timing:
        round_times(args.runs)
        return
    settings = load(args.runs)
    if not settings:
        raise SystemExit(f"no result.json under {args.runs}/<dataset>/<partition>/<run>/")
    out_rows = []
    for setting, groups in settings.items():
        print(f"\n=== {setting}")
        print(f"{'config':42s} n  " + "  ".join(f"{k:>15s}" for k in KEYS))
        for name in sorted(groups):
            g = groups[name]
            cells = {k: fmt([r[k] for r in g.values()]) for k in KEYS}
            print(f"{name:42s} {len(g)}  " + "  ".join(f"{cells[k]:>15s}" for k in KEYS))
            out_rows.append({"setting": setting, "config": name, "n": len(g), **cells})
        if args.ref and args.ref in groups:
            ref = groups[args.ref]
            print(f"--- paired differences vs {args.ref} (negative = lower error): "
                  f"mean +- SD [95% CI] (#seeds lower); single-seed rows show the difference only")
            for name in sorted(groups):
                if name == args.ref:
                    continue
                seeds = sorted(set(groups[name]) & set(ref))
                if not seeds:
                    continue
                cells = []
                for k in DIFF_KEYS:
                    d = np.array([groups[name][s][k] - ref[s][k] for s in seeds])
                    if d.size > 1:
                        hw = T975.get(d.size - 1, 2.0) * d.std(ddof=1) / np.sqrt(d.size)
                        cells.append(f"{k}={d.mean():+.3f}+-{d.std(ddof=1):.3f} "
                                     f"[{d.mean() - hw:+.3f},{d.mean() + hw:+.3f}] ({int((d < 0).sum())}/{d.size})")
                    else:
                        cells.append(f"{k}={d[0]:+.3f}")
                print(f"  {name:40s} " + "  ".join(cells))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["setting", "config", "n"] + KEYS)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
