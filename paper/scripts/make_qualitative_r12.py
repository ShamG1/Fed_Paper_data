"""Trajectory examples selected by the rule in data/qualitative/SELECTION_RULE.md, and the data behind them.

Runs the seed-one checkpoints of FedAvg, CQS, and CQS+FedNova on every test window of the two main settings,
checks the per-window errors against result.json, applies the selection rule, and exports the data of the
trajectory figure. Needs torch, the fedtraj code, the results package, and the processed datasets.

    E:/ANACONDA/envs/g308/python.exe scripts/make_qualitative_r12.py
        [--results E:/SEU_paper/results_fedtraj_20261025] [--data E:/SEU_paper/data] [--code E:/SEU_paper]

Outputs (data/qualitative/)
    qualitative_r12.json         rule file hash, stratum sizes, shares and percentiles of delta, the selected
                                 windows with their errors, and the reproduction checks against result.json
    stratum_errors_ngsim.csv     every test window in Q5 or LC: test index, frame offset, track, client,
    stratum_errors_highd.csv     membership, squared CV error at 5 s, 5 s displacement errors (m) of CV, FedAvg,
                                 CQS, and CQS+FedNova, and delta = e(CQS) - e(FedAvg)
    selected_trajectories_<setting>.csv   history, truth, CV, and the three predictions of the six selected windows
                                 of each setting (m, relative to the last observed position)
    selected_windows_ngsim.npz   raw 80 x 44 frames of the six selected NGSIM windows (to re-run the checkpoints)
    selected_windows_highd.npz   the same for highD (derived from highD: keep out of public repositories)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "qualitative"
SETTINGS = {"ngsim": ("fedtraj_ngsim", "natural"), "highd": ("fedtraj_highd", "motion")}
MODELS = [("fedavg", "FedAvg"), ("cqs", "CQS"), ("cqs-nova", "CQS+FedNova")]
SEED = 1
PCTS = (10, 50, 90)
LC_M = 3.0
BATCH = 4096
HORIZONS = ["1s", "2s", "3s", "4s", "5s"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Store:
    """Window gatherer on a memory-mapped frames.npy (the role of fedtraj.data.WindowStore for inference only)."""

    def __init__(self, frames, device):
        self.frames = frames
        self.device = device
        self._ar = np.arange(80)

    def windows(self, starts):
        idx = np.asarray(starts)[:, None] + self._ar[None, :]
        return torch.from_numpy(np.asarray(self.frames[idx], dtype=np.float32)).to(self.device)


def make_tf(FeatureTransform, store, saved, cv_k, device):
    """FeatureTransform with the standardization and residual scale stored in the checkpoint."""
    tf = FeatureTransform.__new__(FeatureTransform)
    tf.store = store
    tf.cv_k = cv_k
    tf.mean = saved["mean"].to(device)
    tf.std = saved["std"].to(device)
    tf.res_scale = saved["res_scale"].to(device)
    return tf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=r"E:\SEU_paper\results_fedtraj_20261025")
    ap.add_argument("--data", default=r"E:\SEU_paper\data")
    ap.add_argument("--code", default=r"E:\SEU_paper")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.code)))
    from fedtraj.data import HORIZON_STEPS, FeatureTransform
    from fedtraj.model import build_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True          # as in fedtraj.train.setup_torch
    torch.backends.cudnn.allow_tf32 = True
    rule = OUT / "SELECTION_RULE.md"
    report = {"rule_file": rule.name, "rule_sha256": sha256(rule),
              "rule_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rule.stat().st_mtime)),
              "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": torch.cuda.get_device_name(0)
              if dev.type == "cuda" else "cpu", "torch": torch.__version__, "seed": SEED,
              "percentiles": list(PCTS), "lc_threshold_m": LC_M, "settings": {}}
    for key, (ds, part) in SETTINGS.items():
        traj_rows = []
        d = Path(args.data) / ds
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        frames = np.load(d / "frames.npy", mmap_mode="r")
        starts = np.load(d / "win_test.npy")
        wtrack = np.load(d / "wtrack_test.npy")
        client = np.load(d / "tracks.npz")[part][wtrack].astype(np.int64)
        names = meta["partitions"][part]["names"]
        store = Store(frames, dev)
        models, tfs, res, rounds = {}, {}, {}, {}
        for cfg, _ in MODELS:
            run = Path(args.results) / "runs_fedtraj" / ds / part / f"{cfg}_s{SEED}"
            ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)
            m = build_model().to(dev)
            m.load_state_dict(ck["model"])
            m.eval()
            models[cfg], rounds[cfg] = m, int(ck["round"])
            tfs[cfg] = make_tf(FeatureTransform, store, ck["tf"], int(meta["cv_k"]), dev)
            res[cfg] = json.loads((run / "result.json").read_text(encoding="utf-8"))
        n = starts.shape[0]
        e5 = {c: np.empty(n) for c in ["cv"] + [m for m, _ in MODELS]}      # Euclidean error at 5 s (m)
        w5 = {m: np.empty(n) for m, _ in MODELS}                             # squared error at 5 s / 2 coords
        sqh = {m: np.zeros(5) for m, _ in MODELS}
        cv5 = np.empty(n, np.float32)
        lat5 = np.empty(n, np.float32)
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, n, BATCH):
                w = store.windows(starts[s:s + BATCH])
                b = w.shape[0]
                fut, cv = tfs["fedavg"]._targets(w)
                cv5[s:s + b] = (((cv[:, 49] - fut[:, 49]) ** 2).sum(-1) / 2).cpu().numpy()
                e5["cv"][s:s + b] = ((cv[:, 49] - fut[:, 49]) ** 2).sum(-1).sqrt().cpu().numpy()
                lat5[s:s + b] = fut[:, 49, 0].abs().cpu().numpy()
                for cfg, _ in MODELS:
                    tf = tfs[cfg]
                    x = (tf._raw_inputs(w) - tf.mean) / tf.std
                    pred = tf.predict(models[cfg], x, cv)
                    e2 = ((pred - fut)[:, HORIZON_STEPS] ** 2).sum(-1)       # [B, 5], summed over the 2 coords
                    sqh[cfg] += e2.double().sum(0).cpu().numpy()
                    e5[cfg][s:s + b] = e2[:, 4].sqrt().cpu().numpy()
                    w5[cfg][s:s + b] = (e2[:, 4] / 2).cpu().numpy()
        # strata exactly as fedtraj.data.stratify (the test split is below its 200,000-window subsample)
        cv5_t = torch.from_numpy(cv5)
        edges = torch.quantile(cv5_t, torch.tensor([0.2, 0.4, 0.6, 0.8]))
        q5 = (torch.bucketize(cv5_t, edges) == 4).numpy()
        lc = lat5 > LC_M
        # reproduction checks against result.json
        checks = {}
        for cfg, _ in MODELS:
            t = res[cfg]["test"]
            rmse = np.sqrt(sqh[cfg] / (2 * n))
            got = {h: float(v) for h, v in zip(HORIZONS, rmse)}
            got["Q5"] = float(np.sqrt(w5[cfg][q5].mean()))
            got["lateral"] = float(np.sqrt(w5[cfg][lc].mean()))
            ref = {h: float(t[h]) for h in HORIZONS}
            ref["Q5"] = float(res[cfg]["strata"]["Q5"])
            ref["lateral"] = float(res[cfg]["strata"]["lateral"])
            checks[cfg] = {"best_round": rounds[cfg], "recomputed": got, "result_json": ref,
                           "max_abs_diff_m": float(max(abs(got[k] - ref[k]) for k in ref))}
        lat_share_ref = float(res["fedavg"]["strata"]["lateral_share"])
        st = {"n_test": int(n), "q5_edge_m2": float(edges[3]), "n_Q5": int(q5.sum()), "n_LC": int(lc.sum()),
              "n_both": int((q5 & lc).sum()), "lc_share": float(lc.mean()), "lc_share_result_json": lat_share_ref,
              "checks": checks, "inference_s": round(time.time() - t0, 1), "strata": {}}
        picks = []
        for stratum, mask in (("Q5", q5), ("LC", lc)):
            idx = np.flatnonzero(mask)
            delta = e5["cqs"][idx] - e5["fedavg"][idx]
            dnova = e5["cqs-nova"][idx] - e5["fedavg"][idx]
            target = np.percentile(delta, PCTS)
            sel = []
            for p, tv in zip(PCTS, target):
                j = int(np.lexsort((idx, np.abs(delta - tv)))[0])   # nearest to the percentile, then lower index
                k = int(idx[j])
                sel.append({"percentile": p, "target_delta_m": float(tv), "test_index": k,
                            "frame_offset": int(starts[k]), "track": int(wtrack[k]), "client": names[client[k]],
                            "in_Q5": bool(q5[k]), "in_LC": bool(lc[k]), "cv5_m2": float(cv5[k]),
                            "e5_m": {c: float(e5[c][k]) for c in e5}, "delta_m": float(delta[j])})
                picks.append((stratum, p, k))
            st["strata"][stratum] = {
                "n": int(idx.size), "share_cqs_closer": float((delta < 0).mean()),
                "median_delta_m": float(np.median(delta)), "mean_delta_m": float(delta.mean()),
                "percentiles_delta_m": {str(p): float(v) for p, v in zip(PCTS, target)},
                "share_cqsnova_closer": float((dnova < 0).mean()), "median_delta_cqsnova_m": float(np.median(dnova)),
                "selected": sel}
        union = np.flatnonzero(q5 | lc)
        with open(OUT / f"stratum_errors_{key}.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["test_index", "frame_offset", "track", "client", "in_Q5", "in_LC", "cv5_sq_m2",
                         "e5_cv_m", "e5_fedavg_m", "e5_cqs_m", "e5_cqsnova_m", "delta_cqs_minus_fedavg_m"])
            for k in union:
                wr.writerow([int(k), int(starts[k]), int(wtrack[k]), names[client[k]], int(q5[k]), int(lc[k]),
                             f"{cv5[k]:.6g}", f"{e5['cv'][k]:.6g}", f"{e5['fedavg'][k]:.6g}", f"{e5['cqs'][k]:.6g}",
                             f"{e5['cqs-nova'][k]:.6g}", f"{e5['cqs'][k] - e5['fedavg'][k]:.6g}"])
        # trajectories of the selected windows
        ks = np.array([k for _, _, k in picks])
        with torch.no_grad():
            w = store.windows(starts[ks])
            fut, cv = tfs["fedavg"]._targets(w)
            hist = (w[:, :30, 0:2] - w[:, 29:30, 0:2]).cpu().numpy()
            series = {"truth": fut.cpu().numpy(), "cv": cv.cpu().numpy()}
            for cfg, _ in MODELS:
                tf = tfs[cfg]
                series[cfg] = tf.predict(models[cfg], (tf._raw_inputs(w) - tf.mean) / tf.std, cv).cpu().numpy()
        for i, (stratum, p, k) in enumerate(picks):
            for step in range(30):
                traj_rows.append([key, stratum, p, int(k), "history", step - 29, round((step - 29) / 10, 1),
                                  f"{hist[i, step, 0]:.4f}", f"{hist[i, step, 1]:.4f}"])
            for name, arr in series.items():
                for step in range(50):
                    traj_rows.append([key, stratum, p, int(k), name, step + 1, round((step + 1) / 10, 1),
                                      f"{arr[i, step, 0]:.4f}", f"{arr[i, step, 1]:.4f}"])
        np.savez_compressed(OUT / f"selected_windows_{key}.npz", frames=w.cpu().numpy(), test_index=ks,
                            frame_offset=starts[ks], stratum=np.array([s for s, _, _ in picks]),
                            percentile=np.array([p for _, p, _ in picks]),
                            channels=np.array(meta["channels"]))
        with open(OUT / f"selected_trajectories_{key}.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["setting", "stratum", "percentile", "test_index", "series", "step", "t_s", "lat_m",
                         "lon_m"])
            wr.writerows(traj_rows)
        # final positions of the selected windows: where each prediction ends relative to the truth
        for i, (stratum, p, k) in enumerate(picks):
            end = {name: arr[i, 49] for name, arr in series.items()}
            entry = next(s for s in st["strata"][stratum]["selected"] if s["percentile"] == p)
            entry["end_lat_m"] = {name: float(v[0]) for name, v in end.items()}
            entry["end_lon_m"] = {name: float(v[1]) for name, v in end.items()}
            entry["cv_error_lon_share"] = float((end["cv"][1] - end["truth"][1]) ** 2 /
                                                max(((end["cv"] - end["truth"]) ** 2).sum(), 1e-12))
        report["settings"][key] = st
        print(f"{key}: {n} test windows, Q5 {st['n_Q5']}, LC {st['n_LC']}, inference {st['inference_s']} s, "
              f"max |recomputed - result.json| = {max(c['max_abs_diff_m'] for c in checks.values()):.2e} m")
        for stratum, v in st["strata"].items():
            print(f"   {stratum}: n={v['n']} CQS closer on {100 * v['share_cqs_closer']:.1f}%  median delta "
                  f"{v['median_delta_m']:+.3f} m  picks " + ", ".join(
                      f"P{s['percentile']} #{s['test_index']} delta {s['delta_m']:+.2f}" for s in v["selected"]))
    (OUT / "qualitative_r12.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
