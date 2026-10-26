"""Paired comparisons of the checks after revision R16 (``results_r17``) with the definitions of make_stats_r12.py.

Pairing by seed within one setting; a difference is the first configuration minus the second, relative to the mean
of the second on the shared seeds; two-sided 95% t interval; "effect" (marked *) if the interval excludes zero and
all seeds agree in sign.  Configurations marked [old] are the runs of the paper (first server, results_fedtraj_
20261025); the others ran on a second server (results_r17).

Experiment 1, admission timing at fixed visit counts: CQS with rising quotas (0.6, 0.8, 1.0) against the same
ranking with falling quotas (1.0, 0.8, 0.6), both with the BatchNorm statistics re-estimated on all windows of a
client after its local epochs (-bn), eight seeds; three seeds of each without re-estimation.  Experiment 2, a
second backbone: an LSTM encoder-decoder without the CV anchor, FedAvg, CQS, CQS-random, and FedBSS-fixed, five
seeds, with the learning rate chosen on the validation error of FedAvg (seed one; runs_r17/tuned_r17.json).

Usage, from the repository root after unpacking the result archives (only result.json and log.csv are read):
    tar -xzf data/fedtraj_results_20261025_part1.tar.gz        # -> results_fedtraj_20261025/
    tar -xzf data/fedtraj_results_r17_20261026_runs.tar.gz     # -> results_r17/
    python paper/scripts/analyze_r17.py --txt paper/data/r17/analysis_r17.txt --json paper/data/r17/analysis_r17.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import scipy.stats as sps

SETTINGS = [("fedtraj_ngsim", "natural", "NGSIM"), ("fedtraj_highd", "motion", "highD")]
KEYS = ["5s", "mean", "Q1", "Q2", "Q3", "Q4", "Q5", "lateral", "worst"]
LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def load(base):
    """Runs of one setting: {configuration: {seed: row}}."""
    out = {}
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        m = re.match(r"(.+)_s(\d+)$", r["run_id"])
        t = r["test"]
        row = {h: float(t[h]) for h in ("1s", "2s", "3s", "4s", "5s", "mean")}
        row["worst"] = max(v for v, n in zip(t["per_client_5s"], t["per_client_n"]) if n > 0)
        for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "lateral"):
            row[q] = float(r["strata"][q])
        row["best_round"] = int(r["best_round"])
        row["best_val"] = float(r["best_val_5s"])
        curve = [float(c["5s"]) for c in r["val_curve"]]
        row["curve"] = curve
        row["val_rise_rel"] = 100 * (curve[-1] / row["best_val"] - 1)    # final-round minus best validation 5 s
        with open(d / "log.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
        row["visits"] = sum(int(x["visits"]) for x in rows)
        row["wall_min"] = r["wall_s"] / 60
        row["gpu"] = r["gpu"]
        out.setdefault(m.group(1), {})[int(m.group(2))] = row
    return out


def paired(ra, rb, key):
    seeds = sorted(set(ra) & set(rb))
    a = np.array([ra[s][key] for s in seeds])
    b = np.array([rb[s][key] for s in seeds])
    d = a - b
    n = d.size
    out = {"n": n, "seeds": seeds, "d": float(d.mean()), "rel": float(100 * d.mean() / b.mean()),
           "lower": int((d < 0).sum()), "per_seed_rel": [float(100 * x / b.mean()) for x in d]}
    if n > 1:
        hw = float(sps.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
        lo, hi = d.mean() - hw, d.mean() + hw
        out.update(rel_lo=float(100 * lo / b.mean()), rel_hi=float(100 * hi / b.mean()))
        excl = lo > 0 or hi < 0
        out["verdict"] = "effect" if excl and out["lower"] in (0, n) else ("mixed" if excl else "unresolved")
    return out


def fmt_pair(p, cm=False):
    s = f"{p['rel']:+6.2f}%"
    if cm:
        s += f" ({100 * p['d']:+5.1f} cm)"
    if "rel_lo" in p:
        s += f" [{p['rel_lo']:+6.2f}, {p['rel_hi']:+6.2f}] {p['lower']}/{p['n']}"
        s += " *" if p["verdict"] == "effect" else ""
    else:
        s += f" (n={p['n']})"
    return s


def describe(runs, cfg):
    rr = runs[cfg]
    seeds = sorted(rr)
    v = np.array([rr[s]["5s"] for s in seeds])
    m = np.array([rr[s]["mean"] for s in seeds])
    curve = np.mean([rr[s]["curve"] for s in seeds], axis=0)
    sd = f"{v.std(ddof=1):.3f}" if v.size > 1 else "  -  "
    return (f"{cfg:26s} n={len(rr)}  test 5s {v.mean():.3f} +- {sd}  1-5s {m.mean():.3f} | best val "
            f"{np.mean([rr[s]['best_val'] for s in seeds]):.3f}, best rounds {sorted(rr[s]['best_round'] for s in seeds)}"
            f", final {np.mean([rr[s]['val_rise_rel'] for s in seeds]):+.2f}% above best | mean curve: minimum "
            f"{curve.min():.3f} at round {int(curve.argmin()) + 1}, round 5 {curve[4]:.3f}, last {curve[-1]:.3f} | "
            f"visits {np.mean([rr[s]['visits'] for s in seeds]):.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="results_fedtraj_20261025", help="unpacked results of the paper")
    ap.add_argument("--new", default="results_r17", help="unpacked results of the checks after R16")
    ap.add_argument("--txt", default=None, help="also write the printed tables to this file")
    ap.add_argument("--json", default=None, help="write the comparisons and per-run values to this file")
    a = ap.parse_args()
    for p in (Path(a.old) / "runs_fedtraj", Path(a.new) / "runs_r17"):
        if not p.exists():
            raise SystemExit(f"missing {p}: unpack the result archives first (see the usage above)")
    out = {}
    tuned = json.loads((Path(a.new) / "runs_r17" / "tuned_r17.json").read_text(encoding="utf-8"))
    for ds, part, name in SETTINGS:
        old = load(Path(a.old) / "runs_fedtraj" / ds / part)
        new = load(Path(a.new) / "runs_r17" / ds / part)
        res = {"pairs": {}}
        say(f"\n==================== {name} ({ds}/{part})")
        say("-- configurations (second server; [old]: runs of the paper)")
        for cfg in sorted(new):
            say("  " + describe(new, cfg))
        for cfg in ("fedavg", "cqs", "cqs-random", "cqs-anti"):
            if cfg in old:
                say("  " + describe(old, cfg) + "  [old]")

        def pair(label, ra, rb):
            if not ra or not rb or not set(ra) & set(rb):
                return
            ps = {k: paired(ra, rb, k) for k in KEYS}
            res["pairs"][label] = ps
            say(f"  {label:44s} 5s {fmt_pair(ps['5s'], True)} | 1-5s {fmt_pair(ps['mean'])} | Q5 "
                f"{fmt_pair(ps['Q5'])} | LC {fmt_pair(ps['lateral'])}")
            say(f"  {'':44s} Q1..Q4 " + "  ".join(f"{k} {ps[k]['rel']:+.2f}%" for k in ("Q1", "Q2", "Q3", "Q4"))
                + f" | worst client {fmt_pair(ps['worst'])}")
            say(f"  {'':44s} per seed 5s (%): " + " ".join(f"{x:+.2f}" for x in ps["5s"]["per_seed_rel"]))

        g = lambda d, c: d.get(c, {})  # noqa: E731
        say("-- Experiment 1: admission timing at fixed visit counts")
        pair("CQS+BN - CQS-desc+BN", g(new, "cqs-bn"), g(new, "cqs-desc-bn"))
        pair("CQS-desc - CQS-desc+BN (re-estimation, desc)", g(new, "cqs-desc"), g(new, "cqs-desc-bn"))
        pair("CQS - CQS+BN (re-estimation, CQS)", g(new, "cqs"), g(new, "cqs-bn"))
        pair("CQS - CQS-desc (no re-estimation)", g(new, "cqs"), g(new, "cqs-desc"))
        pair("CQS new - CQS old (same seeds, two servers)", g(new, "cqs"), g(old, "cqs"))
        pair("CQS+BN new - CQS old", g(new, "cqs-bn"), g(old, "cqs"))
        pair("CQS+BN new - FedAvg old", g(new, "cqs-bn"), g(old, "fedavg"))
        pair("CQS-desc+BN new - FedAvg old", g(new, "cqs-desc-bn"), g(old, "fedavg"))
        pair("CQS-desc+BN new - CQS-random old", g(new, "cqs-desc-bn"), g(old, "cqs-random"))
        pair("CQS old - FedAvg old (paper)", g(old, "cqs"), g(old, "fedavg"))
        pair("CQS old - CQS-random old (paper)", g(old, "cqs"), g(old, "cqs-random"))
        say("-- Experiment 2: LSTM backbone without the CV anchor")
        pilot = {c: v[1] for c, v in new.items() if c.startswith("fedavg-lstm") and 1 in v}
        say("  pilot, FedAvg-LSTM seed 1, best validation 5 s: " + "  ".join(
            f"{c}: {r['best_val']:.4f} (round {r['best_round']})" for c, r in sorted(pilot.items())))
        lr = tuned[f"{ds}/{part}"]["lstm_lr"]
        suf = "-lstm" if float(lr) == 5e-4 else f"-lstm-lr{float(lr):g}"
        sel = {k: g(new, k + suf) for k in ("fedavg", "cqs", "fedbss", "cqs-random")}
        res["lstm_lr"] = lr
        say(f"  selected learning rate {lr} (runs_r17/tuned_r17.json)")
        pair("LSTM: CQS - FedAvg", sel["cqs"], sel["fedavg"])
        pair("LSTM: CQS - CQS-random", sel["cqs"], sel["cqs-random"])
        pair("LSTM: CQS - FedBSS-fixed", sel["cqs"], sel["fedbss"])
        pair("LSTM: CQS-random - FedAvg", sel["cqs-random"], sel["fedavg"])
        pair("LSTM: FedBSS-fixed - FedAvg", sel["fedbss"], sel["fedavg"])
        pair("LSTM FedAvg - FedAvg old (other backbone)", sel["fedavg"], g(old, "fedavg"))
        res["runs"] = {c: {s: {k: v for k, v in r.items() if k != "curve"} for s, r in rr.items()}
                       for c, rr in new.items()}
        res["mean_val_curves"] = {c: np.mean([r["curve"] for r in rr.values()], axis=0).round(5).tolist()
                                  for c, rr in new.items()}
        out[f"{ds}/{part}"] = res
    if a.txt:
        Path(a.txt).parent.mkdir(parents=True, exist_ok=True)
        Path(a.txt).write_text("\n".join(LINES).lstrip("\n") + "\n", encoding="utf-8")
        print(f"\nwrote {a.txt}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
