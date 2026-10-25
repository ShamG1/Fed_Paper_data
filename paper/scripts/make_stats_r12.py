"""Compute every reported number of the R12 manuscript from the fedtraj results package.

    python scripts/make_stats_r12.py [--results DIR] [--data DIR] [--recompute-difficulty]

Inputs
    --results  results_fedtraj_20261025 (runs_fedtraj/<dataset>/<partition>/<run>/{result.json,log.csv},
               runs_timing/...); default E:/SEU_paper/results_fedtraj_20261025
    --data     directory holding fedtraj_ngsim/ and fedtraj_highd/ (frames.npy, tracks.npz, win_train.npy,
               wtrack_train.npy, meta.json); only read for the model-free client statistics
Outputs (in data/)
    stats_r12.json       all statistics quoted in the paper; read by make_tables_r12.py and check_numbers_r12.py
    figdata_r12.json     curves and scatter points for the figures (not used by the number check)
    difficulty_r12.json  cache of the model-free client statistics (recomputed with --recompute-difficulty)

Conventions (pre-specified in fedtraj/README.md, section 0.4)
    metric      test RMSE at the checkpoint with the lowest validation 5 s RMSE
    pairing     by seed within one setting; difference = first minus second configuration
    summary     mean, sample SD, two-sided 95% t interval, number of seeds on which the first is lower;
                relative values refer to the mean of the second configuration on the shared seeds
    verdict     "effect" if the interval excludes zero and every seed has the same sign, else "unresolved"
                ("mixed" if the interval excludes zero but the signs differ)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
DEF_RESULTS = Path(r"E:\SEU_paper\results_fedtraj_20261025")
DEF_DATA = Path(r"E:\SEU_paper\data")
WARMUP = 5                       # CQS warm-up rounds; diagnostics use rounds 6..30
HORIZONS = ["1s", "2s", "3s", "4s", "5s", "mean"]
STRATA = ["Q1", "Q2", "Q3", "Q4", "Q5", "lateral"]

SETTINGS = {
    "ngsim": ("fedtraj_ngsim", "natural"),
    "highd": ("fedtraj_highd", "motion"),
    "ngsim_motion": ("fedtraj_ngsim", "motion"),
    "highd_rec": ("fedtraj_highd", "recording"),
}
MAIN = ("ngsim", "highd")

# paired comparisons in both main settings (first, second)
MAIN_PAIRS = [
    # pre-specified primary comparisons (eight seeds)
    ("cqs-udr", "fedavg"), ("cqs", "fedavg"), ("cqs", "cqs-random"), ("cqs-udr", "cqs"), ("cqs-udr", "fedprox"),
    # scheduler against the other client- and server-side rules
    ("cqs", "fedprox"), ("cqs", "fednova"), ("cqs", "fedlc"), ("cqs", "fedbss"), ("cqs", "fedbss-mean"),
    ("cqs", "cqs-anti"),
    ("cqs-random", "fedavg"), ("cqs-anti", "fedavg"), ("fedprox", "fedavg"), ("fednova", "fedavg"),
    ("fedlc", "fedavg"), ("fedbss", "fedavg"), ("fedbss-mean", "fedavg"), ("udr", "fedavg"),
    # CQS added to existing rules (added after the main results)
    ("cqs-nova", "fednova"), ("cqs-prox", "fedprox"), ("cqs-nova", "cqs"), ("cqs-prox", "cqs"),
    ("cqs-nova", "fedavg"), ("cqs-prox", "fedavg"),
    # server layer with CQS fixed
    ("cqs-static", "cqs"), ("cqs-udr", "cqs-static"), ("cqs-udrfs", "cqs"), ("cqs-udr", "cqs-udrfs"),
    ("cqs-nova", "cqs-udr"), ("cqs-prox", "cqs-udr"), ("cqs-nova", "cqs-static"),
    ("cqs-udr", "fednova"), ("cqs-udr", "fedlc"), ("cqs-udr", "fedbss"), ("cqs-udr", "fedbss-mean"),
    ("cqs-udr", "udr"),
]
HALF_PAIRS = [("cqs-frac0.5", "fedavg-frac0.5"), ("fedavg-udr-frac0.5", "fedavg-frac0.5"),
              ("cqs-udr-frac0.5", "fedavg-frac0.5"), ("cqs-udr-frac0.5", "cqs-frac0.5")]
NGSIM_MOTION_PAIRS = [("cqs", "fedavg"), ("cqs", "cqs-random"), ("cqs-random", "fedavg"), ("udr", "fedavg"),
                      ("cqs-udr", "fedavg"), ("cqs-udr", "cqs")]
HIGHD_REC_PAIRS = [("cqs", "fedavg"), ("udr", "fedavg"), ("cqs-udr", "fedavg"), ("cqs-udr", "cqs"),
                   ("fednova", "fedavg"), ("cqs", "fednova"), ("cqs-udr", "fednova")]
PACING_PAIRS = [("cqs-cosine-E5", "cqs-E5")]
PAIR_KEYS = ["5s", "mean", "1s", "Q5", "lateral", "worst"]
SENS = ["cqs-f0.4-udr", "cqs-f0.8-udr", "cqs-w0-udr", "cqs-w10-udr", "cqs-udr-t0.5", "cqs-udr-t2",
        "cqs-udr-b0.7-1.3", "cqs-udr-b0.3-1.7"]
SENS_REPS = ["cqs-udr", "cqs-udr-rep2", "cqs-udr-rep3"]


def canonical(prefix):
    """Run-id prefix -> configuration key used in the paper scripts."""
    prefix = re.sub(r"^cqs-static[0-9.]+", "cqs-static", prefix)
    prefix = re.sub(r"^fedprox-mu0\.01$", "fedprox", prefix)
    prefix = re.sub(r"^cqs-prox0\.01$", "cqs-prox", prefix)
    prefix = re.sub(r"^fedavg-udr", "udr", prefix)
    prefix = re.sub(r"-frac0\.1-r100$", "", prefix)        # highD recording runs all use 10% / 100 rounds
    if prefix == "udr-frac0.5":
        prefix = "fedavg-udr-frac0.5"
    return prefix


def read_log(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        r["extra"] = json.loads(r["extra"]) if r.get("extra") else {}
        rows.append(r)
    return rows


def load_setting(results, ds, part):
    base = results / "runs_fedtraj" / ds / part
    out = {}
    for d in sorted(base.iterdir()):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        m = re.match(r"(.+)_s(\d+)$", r["run_id"])
        cfg, seed = canonical(m.group(1)), int(m.group(2))
        t = r["test"]
        row = {h: float(t[h]) for h in HORIZONS}
        row["per_client"] = [float(v) for v in t["per_client_5s"]]
        row["per_client_n"] = [int(v) for v in t["per_client_n"]]
        row["worst"] = max(v for v, n in zip(row["per_client"], row["per_client_n"]) if n > 0)
        for q in STRATA:
            row[q] = float(r["strata"][q])
        row["lateral_share"] = float(r["strata"]["lateral_share"])
        row["best_round"] = int(r["best_round"])
        row["best_val"] = float(r["best_val_5s"])
        row["last_test_5s"] = float(r["test_last_round"]["5s"])
        curve = [float(c["5s"]) for c in r["val_curve"]]
        row["val_curve"] = curve
        row["val_last10"] = float(np.mean(curve[-10:]))
        row["val_rise"] = float(curve[-1] - row["best_val"])          # final-round minus best validation 5 s
        row["val_rise_rel"] = float(100 * (curve[-1] / row["best_val"] - 1))
        row["train_counts"] = r["train_counts"]
        row["args"] = r["args"]
        log = read_log(d / "log.csv")
        row["visits"] = int(sum(int(x["visits"]) for x in log))
        row["log"] = log
        out.setdefault(cfg, {})[seed] = row
    return out


def summary(runs, key):
    v = np.array([runs[s][key] for s in sorted(runs)], dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "max": float(v.max())}


def paired(ra, rb, key):
    seeds = sorted(set(ra) & set(rb))
    a = np.array([ra[s][key] for s in seeds], dtype=float)
    b = np.array([rb[s][key] for s in seeds], dtype=float)
    d = a - b
    n = d.size
    out = {"n": n, "seeds": seeds, "d": float(d.mean()), "per_seed": [float(x) for x in d],
           "lower": int((d < 0).sum()), "ref_mean": float(b.mean()), "rel": float(100 * d.mean() / b.mean())}
    if n > 1:
        sd = float(d.std(ddof=1))
        hw = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        lo, hi = out["d"] - hw, out["d"] + hw
        out.update(sd=sd, lo=lo, hi=hi, rel_lo=100 * lo / b.mean(), rel_hi=100 * hi / b.mean())
        excl = lo > 0 or hi < 0
        same = out["lower"] in (0, n)
        out["verdict"] = "effect" if excl and same else ("mixed" if excl else "unresolved")
    return out


def pair_block(runs, pairs, keys=PAIR_KEYS):
    out = {}
    for a, b in pairs:
        if a in runs and b in runs and set(runs[a]) & set(runs[b]):
            out[f"{a}|{b}"] = {k: paired(runs[a], runs[b], k) for k in keys}
    return out


def config_block(runs, keys=HORIZONS + STRATA + ["worst", "best_val", "val_last10", "best_round", "visits",
                                                  "val_rise", "val_rise_rel"]):
    out = {}
    for cfg, rr in runs.items():
        out[cfg] = {k: summary(rr, k) for k in keys}
        out[cfg]["seeds"] = sorted(rr)
        out[cfg]["best_rounds"] = [rr[s]["best_round"] for s in sorted(rr)]
        ke = [np.mean([float(x["extra"]["keff"]) for x in rr[s]["log"] if "keff" in x["extra"]])
              for s in sorted(rr) if any("keff" in x["extra"] for x in rr[s]["log"])]
        if ke:
            out[cfg]["keff_all"] = float(np.mean(ke))
    return out


def post(log):
    return [x for x in log if int(x["round"]) > WARMUP]


def corr_within(runs, xkey, ykey, seeds=range(1, 6)):
    """Pearson r between two per-client logged quantities over client-rounds after warm-up, per seed."""
    rs = []
    for s in seeds:
        if s not in runs:
            continue
        x, y = [], []
        for r in post(runs[s]["log"]):
            e = r["extra"]
            x += list(map(float, e[xkey]))
            y += list(map(float, e[ykey]))
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    return {"mean": float(np.mean(rs)), "min": float(np.min(rs)), "max": float(np.max(rs)), "n": len(rs)}


def server_diagnostics(runs, n_train):
    n = np.array(n_train, float)
    p = n / n.sum()
    out = {"client_share": p.tolist(), "keff_fedavg": float(1 / np.sum(p ** 2)),
           "k_over_keff": float(len(n) * np.sum(p ** 2))}
    # update norm and UDR deviation against the number of local steps
    out["r_norm_steps"] = {c: corr_within(runs[c], "delta_norm", "steps") for c in ("fedavg", "cqs", "cqs-udr")
                           if c in runs}
    out["r_dev_steps"] = corr_within(runs["cqs-udr"], "udr_dev", "steps")
    # UDR multipliers and influence ratios alpha/p (CQS+UDR, seeds 1-5, rounds 6-30)
    S, A, cl = [], [], []
    between_share = []
    for s in range(1, 6):
        logs = post(runs["cqs-udr"][s]["log"])
        sm = np.array([list(map(float, r["extra"]["udr_s"])) for r in logs])           # [rounds, K]
        al = np.array([list(map(float, r["extra"]["alpha"])) for r in logs])
        pp = np.array([list(map(float, r["extra"]["p"])) for r in logs])
        S.append(sm)
        A.append(al / pp)
        ls = np.log(sm)
        tot = ls.var()
        between_share.append(float(ls.mean(0).var() / tot) if tot > 0 else 1.0)
    S, A = np.concatenate(S), np.concatenate(A)
    ratio_udr = A.mean(0)
    beta = runs["cqs-static"][1]["args"]["static_beta"]
    w = n ** beta / np.sum(n ** beta)
    ratio_static = w / p
    at_max = (S >= 1.5 - 1e-6).mean(0)                     # per client: share of its rounds at s_max
    small = p < 0.06
    out["udr"] = {
        "alpha_over_p": ratio_udr.tolist(), "alpha_over_p_min": float(ratio_udr.min()),
        "alpha_over_p_max": float(ratio_udr.max()),
        "share_at_smax_per_client": at_max.tolist(),
        "small_clients_lt6pct": int(small.sum()),
        "share_at_smax_small_min": float(at_max[small].min()) if small.any() else None,
        "share_at_smax_small_max": float(at_max[small].max()) if small.any() else None,
        "share_at_smax_others_max": float(at_max[~small].max()) if (~small).any() else None,
        "share_at_smax": float(np.mean(S >= 1.5 - 1e-6)), "share_at_smin": float(np.mean(S <= 0.5 + 1e-6)),
        "within_client_sd_s": float(np.mean(S.std(0))),
        "between_client_var_share_log_s": float(np.mean(between_share)),
        "r_log_ratio_log_n": float(np.corrcoef(np.log(ratio_udr), np.log(n))[0, 1]),
    }
    out["static"] = {"beta": beta, "alpha_over_p": ratio_static.tolist(),
                     "alpha_over_p_min": float(ratio_static.min()), "alpha_over_p_max": float(ratio_static.max()),
                     "keff": float(1 / np.sum(w ** 2))}
    out["udr_vs_static_max_abs_diff"] = float(np.max(np.abs(ratio_udr - ratio_static)))
    # FedNova: the factor tau_eff / tau_i that multiplies p_i (CQS+FedNova and FedNova, rounds 6-30)
    for c in ("cqs-nova", "fednova"):
        if c not in runs:
            continue
        F = []
        for s in range(1, 6):
            for r in post(runs[c][s]["log"]):
                tau = np.array(list(map(float, r["extra"]["steps"])))
                F.append(float(r["extra"]["tau_eff"]) / tau)
        F = np.array(F).mean(0)
        out[f"{c}_factor"] = {"tau_eff_over_tau": F.tolist(), "min": float(F.min()), "max": float(F.max()),
                              "sum_p_factor": float(np.dot(p, F))}
    # leave-one-out alignment of the updates after step normalization (FedNova runs): is there a stable,
    # client-specific signal left for a deviation-based weight?
    for c in ("fednova", "cqs-nova", "fedavg"):
        if c not in runs:
            continue
        betw, within, split = [], [], []
        for s in range(1, 6):
            C = np.array([list(map(float, r["extra"]["cos_loo"])) for r in post(runs[c][s]["log"])])
            m = C.mean(0)
            betw.append(float(m.std(ddof=1)))
            within.append(float(np.mean(C.std(0, ddof=1))))
            split.append(float(np.corrcoef(C[0::2].mean(0), C[1::2].mean(0))[0, 1]))
        out[f"{c}_cos_loo"] = {"between_sd": float(np.mean(betw)), "within_sd": float(np.mean(within)),
                               "split_half_r": float(np.mean(split)),
                               "r_mean_cos_log_n": None}
        # correlation of the client-mean alignment with client size, pooled over seeds
        ms = []
        for s in range(1, 6):
            C = np.array([list(map(float, r["extra"]["cos_loo"])) for r in post(runs[c][s]["log"])])
            ms.append(C.mean(0))
        out[f"{c}_cos_loo"]["r_mean_cos_log_n"] = float(np.corrcoef(np.mean(ms, 0), np.log(n))[0, 1])
    return out


def scatter_data(runs, cfg, seeds=range(1, 6)):
    pts = []
    for s in seeds:
        for r in post(runs[cfg][s]["log"]):
            e = r["extra"]
            for i, c in enumerate(e["cohort"]):
                pts.append([int(c), float(e["steps"][i]), float(e["delta_norm"][i]),
                            float(e["udr_dev"][i]) if "udr_dev" in e else None])
    return pts


# ----------------------------------------------------------------------------- model-free client statistics
def cv_losses(xy, starts, chunk=20000):
    """Per-window CV loss: mean over 50 future steps x 2 coordinates of the squared CV error (m^2)."""
    k = np.arange(1, 51, dtype=np.float64)[None, :, None]
    loss = np.empty(starts.shape[0])
    lat5 = np.empty(starts.shape[0])
    for s in range(0, starts.shape[0], chunk):
        st = starts[s:s + chunk]
        idx = st[:, None] + np.arange(80)[None, :]
        w = xy[idx]                                                    # [B, 80, 2]
        last = w[:, 29]
        inc = (last - w[:, 24]) / 5.0
        true = w[:, 30:] - last[:, None, :]
        err = inc[:, None, :] * k - true
        loss[s:s + chunk] = (err ** 2).mean(axis=(1, 2))
        lat5[s:s + chunk] = np.abs(true[:, 49, 0])
    return loss, lat5


def difficulty(data_root, ds, parts):
    d = data_root / ds
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    frames = np.load(d / "frames.npy", mmap_mode="r")
    xy = np.asarray(frames[:, 0:2], dtype=np.float64)
    tracks = np.load(d / "tracks.npz")
    st = np.load(d / "win_train.npy")
    owner = np.load(d / "wtrack_train.npy")
    loss, lat5 = cv_losses(xy, st)
    out = {"n_windows": int(st.shape[0]), "lateral_share_train": float(np.mean(lat5 > 3.0))}
    for part in parts:
        lab = tracks[part][owner].astype(np.int64)
        K = len(meta["partitions"][part]["names"])
        rows = []
        for c in range(K):
            lc = loss[lab == c]
            q = np.percentile(lc, [5, 25, 50, 75, 95])
            top = np.sort(lc)[::-1][:int(np.ceil(0.2 * lc.size))]
            rows.append({"n": int(lc.size), "q": q.tolist(), "mean": float(lc.mean()),
                         "p95_over_p5": float(q[4] / q[0]), "top20_share": float(top.sum() / lc.sum()),
                         "lateral_share": float(np.mean(lat5[lab == c] > 3.0))})
        med = np.array([r["q"][2] for r in rows])
        ratio = np.array([r["p95_over_p5"] for r in rows])
        top = np.array([r["top20_share"] for r in rows])
        size = np.array([r["n"] for r in rows], float)
        out[part] = {"names": meta["partitions"][part]["names"], "clients": rows,
                     "mean_v_lon": meta["partitions"][part]["mean_v_lon"],
                     "p95_over_p5_min": float(ratio.min()), "p95_over_p5_max": float(ratio.max()),
                     "median_max_over_min": float(med.max() / med.min()),
                     "median_min": float(med.min()), "median_max": float(med.max()),
                     "top20_share_min": float(top.min()), "top20_share_max": float(top.max()),
                     "spearman_size_median": float(sps.spearmanr(size, med)[0])}
    return out


def dataset_facts(data_root, ds):
    meta = json.loads((data_root / ds / "meta.json").read_text(encoding="utf-8"))
    out = {"n_tracks": meta["n_tracks"], "n_frames": meta["n_frames"], "strides": meta["strides"],
           "loss_scale": meta["loss_scale"], "cv": meta["cv_rmse"], "partitions": {}}
    for part, info in meta["partitions"].items():
        c = info["counts"]
        n = np.array(c["train"], float)
        out["partitions"][part] = {"K": len(info["names"]), "train": c["train"], "n_train": int(n.sum()),
                                   "n_valid": int(sum(c["valid"])), "n_test": int(sum(c["test"])),
                                   "share_min": float(n.min() / n.sum()), "share_max": float(n.max() / n.sum()),
                                   "keff_size": float(1 / np.sum((n / n.sum()) ** 2)),
                                   "mean_v_lon": info["mean_v_lon"]}
    if "periods" in meta:
        out["vehicles"] = int(sum(r["vehicles"] for r in meta["periods"]))
    if "recordings" in meta:
        out["vehicles"] = int(sum(r["vehicles"] for r in meta["recordings"]))
        out["n_recordings"] = len(meta["recordings"])
    return out


def timing(results):
    out = {}
    for f in sorted((results / "runs_timing").glob("*/*/*/log.csv")):
        rows = read_log(f)
        t = [float(r["time_s"]) for r in rows[1:]]
        v = [float(r["visits"]) for r in rows[1:]]
        out[f.parts[-2]] = {"s_per_round": float(np.mean(t)), "visits_per_round": float(np.mean(v)),
                            "rounds_timed": len(t)}
    fa, cq = out["fedavg-r3_s1"], out["cqs-w0-r3_s1"]
    out["derived"] = {"cqs_time_saving_pct": 100 * (1 - cq["s_per_round"] / fa["s_per_round"]),
                      "cqs_visits_ratio": cq["visits_per_round"] / fa["visits_per_round"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(DEF_RESULTS))
    ap.add_argument("--data", default=str(DEF_DATA))
    ap.add_argument("--recompute-difficulty", action="store_true")
    args = ap.parse_args()
    results, data_root = Path(args.results), Path(args.data)
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    cache = out_dir / "difficulty_r12.json"
    if cache.exists() and not args.recompute_difficulty:
        diff = json.loads(cache.read_text(encoding="utf-8"))
    else:
        diff = {"fedtraj_ngsim": difficulty(data_root, "fedtraj_ngsim", ["natural", "motion"]),
                "fedtraj_highd": difficulty(data_root, "fedtraj_highd", ["motion", "recording"])}
        cache.write_text(json.dumps(diff, indent=1), encoding="utf-8")

    S = {"datasets": {ds: dataset_facts(data_root, ds) for ds in ("fedtraj_ngsim", "fedtraj_highd")},
         "difficulty": {}, "settings": {}, "timing": timing(results)}
    for ds, d in diff.items():
        S["difficulty"][ds] = {p: {k: v for k, v in d[p].items() if k not in ("clients", "names", "mean_v_lon")}
                               for p in d if isinstance(d[p], dict)}
        S["difficulty"][ds]["lateral_share_train"] = d["lateral_share_train"]
    F = {"difficulty": diff, "settings": {}}

    for key, (ds, part) in SETTINGS.items():
        runs = load_setting(results, ds, part)
        st = {"configs": config_block(runs)}
        fd = {}
        if key in MAIN:
            st["pairs"] = pair_block(runs, MAIN_PAIRS)
            st["half"] = pair_block(runs, HALF_PAIRS)
            st["pacing"] = pair_block(runs, PACING_PAIRS) if key == "ngsim" else {}
            n_train = runs["fedavg"][1]["train_counts"]
            st["lateral_share_test"] = runs["fedavg"][1]["lateral_share"]
            st["server"] = server_diagnostics(runs, n_train)
            # gap to the centralized model on seeds 1-5
            gaps = {}
            for cfg in ("fedavg", "fedprox", "fednova", "fedlc", "fedbss", "fedbss-mean", "cqs-random", "cqs",
                        "cqs-udr", "cqs-static", "cqs-nova", "cqs-prox", "udr"):
                if cfg not in runs:
                    continue
                seeds = sorted(set(runs[cfg]) & set(runs["central"]))
                g = [100 * (runs[cfg][s]["5s"] / runs["central"][s]["5s"] - 1) for s in seeds]
                gaps[cfg] = {"n": len(g), "mean": float(np.mean(g))}
            for cfg in gaps:
                gaps[cfg]["closed_vs_fedavg"] = float(100 * (1 - gaps[cfg]["mean"] / gaps["fedavg"]["mean"]))
            st["gap_central"] = gaps
            # relative change of every stratum against FedAvg (paired means), for the figure and the text
            strat = {}
            for cfg in ("cqs", "cqs-random", "cqs-anti", "cqs-udr", "cqs-nova", "fednova", "fedprox", "fedbss"):
                if cfg in runs:
                    strat[cfg] = {q: paired(runs[cfg], runs["fedavg"], q) for q in STRATA + ["5s"]}
            st["strata_vs_fedavg"] = strat
            # visits relative to FedAvg (sum over the 30 rounds, seed 1)
            st["visits_rel"] = {cfg: runs[cfg][1]["visits"] / runs["fedavg"][1]["visits"]
                                for cfg in ("cqs", "cqs-random", "cqs-anti", "fedbss", "fedbss-mean", "cqs-udr",
                                            "cqs-nova", "cqs-prox") if cfg in runs and 1 in runs[cfg]}
            # fewer visits of CQS than the FedBSS adaptations (%)
            st["visits_saving_pct"] = {cfg: 100 * (1 - st["visits_rel"]["cqs"] / st["visits_rel"][cfg])
                                       for cfg in ("fedbss", "fedbss-mean")}
            # pilot: FedProx coefficient grid on seed 1 (validation), and the validation 5 s RMSE of the pilot
            pil = {}
            for cfg in ("fedavg", "cqs", "cqs-udr", "udr", "cqs-random"):
                pil[cfg] = {"best_val": runs[cfg][1]["best_val"], "val_last10": runs[cfg][1]["val_last10"]}
            for mu in ("0.0001", "0.001", "0.01", "0.1", "1"):
                cfg = "fedprox" if mu == "0.01" else f"fedprox-mu{mu}"
                pil[f"fedprox_mu{mu}"] = {"best_val": runs[cfg][1]["best_val"],
                                          "val_last10": runs[cfg][1]["val_last10"]}
            st["pilot"] = pil
            # sensitivity (seed 1, CQS+UDR)
            sens = {c: {"test_5s": runs[c][1]["5s"], "best_val": runs[c][1]["best_val"],
                        "test_mean": runs[c][1]["mean"]} for c in SENS + SENS_REPS if c in runs}
            reps = [runs[c][1] for c in SENS_REPS if c in runs]
            sens["band"] = {"test_5s_min": min(r["5s"] for r in reps), "test_5s_max": max(r["5s"] for r in reps),
                            "best_val_min": min(r["best_val"] for r in reps),
                            "best_val_max": max(r["best_val"] for r in reps)}
            sens["band"]["test_5s_range"] = sens["band"]["test_5s_max"] - sens["band"]["test_5s_min"]
            sens["band"]["best_val_range"] = sens["band"]["best_val_max"] - sens["band"]["best_val_min"]
            st["sensitivity"] = sens
            # figure data
            fd["val_curves"] = {}
            for cfg in ("fedavg", "cqs-random", "cqs", "cqs-udr", "fednova", "cqs-nova", "fedprox", "central"):
                if cfg in runs:
                    C = np.array([runs[cfg][s]["val_curve"] for s in sorted(runs[cfg])])
                    fd["val_curves"][cfg] = {"mean": C.mean(0).tolist(), "sd": C.std(0, ddof=1).tolist(),
                                             "n": int(C.shape[0])}
            fd["scatter"] = {c: scatter_data(runs, c) for c in ("fedavg", "cqs", "cqs-udr") if c in runs}
            fd["n_train"] = n_train
            fd["per_client"] = {cfg: np.mean([runs[cfg][s]["per_client"] for s in sorted(runs[cfg])], 0).tolist()
                                for cfg in ("fedavg", "cqs", "cqs-udr", "cqs-nova", "central") if cfg in runs}
        elif key == "ngsim_motion":
            st["pairs"] = pair_block(runs, NGSIM_MOTION_PAIRS)
        elif key == "highd_rec":
            st["pairs"] = pair_block(runs, HIGHD_REC_PAIRS)
        S["settings"][key] = st
        F["settings"][key] = fd
        print(f"{key}: {len(runs)} configurations, {sum(len(v) for v in runs.values())} runs")

    (out_dir / "stats_r12.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    (out_dir / "figdata_r12.json").write_text(json.dumps(F), encoding="utf-8")
    print("wrote", out_dir / "stats_r12.json", "and figdata_r12.json")


if __name__ == "__main__":
    main()
