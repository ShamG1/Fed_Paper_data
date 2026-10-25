"""Check the trajectory examples without the processed datasets.

    python scripts/check_qualitative_r12.py
    python scripts/check_qualitative_r12.py --results results_fedtraj_20261025 --code ../code    # also re-run (torch)

1. Selection: recomputes the selected windows from data/qualitative/stratum_errors_<setting>.csv (every Q5 and
   lane-change test window with its 5 s errors) by the rule of SELECTION_RULE.md, and compares them with the
   windows listed in qualitative_r12.json.
2. Raw windows: recomputes the history, the true future, and the constant-velocity prediction of every selected
   window from its raw frames (selected_windows_<setting>.npz) and compares them with
   selected_trajectories_<setting>.csv, the data drawn in the trajectory figure.
3. Predictions (with --results and --code; needs torch): runs the seed-one checkpoints of FedAvg, CQS, and
   CQS+FedNova on the raw frames and compares the predicted positions with the same file.
Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "data" / "qualitative"
SETTINGS = {"ngsim": ("fedtraj_ngsim", "natural"), "highd": ("fedtraj_highd", "motion")}
MODELS = ("fedavg", "cqs", "cqs-nova")
PCTS = (10, 50, 90)
TOL_M = 0.01          # predictions: the figure data are rounded to 0.1 mm; GPU (TF32) and CPU arithmetic differ


def figure_data(key):
    """{(stratum, percentile): {series: array [steps, 2] of (lat, lon)}} from selected_trajectories_<key>.csv."""
    out = {}
    with open(Q / f"selected_trajectories_{key}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["stratum"], int(r["percentile"])), {}).setdefault(r["series"], []).append(
                (float(r["lat_m"]), float(r["lon_m"])))
    return {k: {s: np.array(v) for s, v in d.items()} for k, d in out.items()}


def check_selection(key, report):
    rows = list(csv.DictReader(open(Q / f"stratum_errors_{key}.csv", encoding="utf-8")))
    ok = True
    for stratum, col in (("Q5", "in_Q5"), ("LC", "in_LC")):
        sub = [r for r in rows if r[col] == "1"]
        idx = np.array([int(r["test_index"]) for r in sub])
        delta = np.array([float(r["e5_cqs_m"]) - float(r["e5_fedavg_m"]) for r in sub])
        listed = {s["percentile"]: s["test_index"] for s in report["settings"][key]["strata"][stratum]["selected"]}
        for p, target in zip(PCTS, np.percentile(delta, PCTS)):
            got = int(idx[np.lexsort((idx, np.abs(delta - target)))[0]])
            ok &= got == listed[p]
            print(f"  {key} {stratum} P{p}: window {got} ({'matches' if got == listed[p] else 'differs from'} "
                  f"qualitative_r12.json)")
        n_ok = len(sub) == report["settings"][key]["strata"][stratum]["n"]
        ok &= n_ok
        print(f"  {key} {stratum}: {len(sub)} windows in the stratum ({'as listed' if n_ok else 'COUNT DIFFERS'})")
    return ok


def check_raw(key, fig):
    z = np.load(Q / f"selected_windows_{key}.npz")
    w = z["frames"].astype(np.float64)
    worst = 0.0
    for i in range(w.shape[0]):
        k = (str(z["stratum"][i]), int(z["percentile"][i]))
        last = w[i, 29, 0:2]
        series = {"history": w[i, :30, 0:2] - last, "truth": w[i, 30:, 0:2] - last,
                  "cv": ((last - w[i, 24, 0:2]) / 5.0)[None, :] * np.arange(1, 51)[:, None]}
        for s, a in series.items():
            worst = max(worst, float(np.abs(a - fig[k][s]).max()))
    print(f"  {key}: history, truth, and CV of {w.shape[0]} windows recomputed from the raw frames, "
          f"largest difference {worst:.1e} m")
    return worst < 1e-3


def check_predictions(key, fig, results, code):
    import torch
    sys.path.insert(0, str(code))
    from fedtraj.data import FeatureTransform
    from fedtraj.model import build_model

    ds, part = SETTINGS[key]
    z = np.load(Q / f"selected_windows_{key}.npz")
    w = torch.from_numpy(z["frames"].astype(np.float32))
    meta = ROOT / "dataset_meta" / ds / "meta.json"          # in the GitHub repository; else the code constant
    if meta.exists():
        cv_k = int(json.loads(meta.read_text(encoding="utf-8"))["cv_k"])
    else:
        from fedtraj.layout import CV_K as cv_k
    worst = 0.0
    for cfg in MODELS:
        ck = torch.load(Path(results) / "runs_fedtraj" / ds / part / f"{cfg}_s1" / "best.pt", map_location="cpu",
                        weights_only=False)
        model = build_model()
        model.load_state_dict(ck["model"])
        model.eval()
        tf = FeatureTransform.__new__(FeatureTransform)
        tf.cv_k, tf.mean, tf.std, tf.res_scale = cv_k, ck["tf"]["mean"], ck["tf"]["std"], ck["tf"]["res_scale"]
        with torch.no_grad():
            fut, cv = tf._targets(w)
            pred = tf.predict(model, (tf._raw_inputs(w) - tf.mean) / tf.std, cv).numpy()
        for i in range(pred.shape[0]):
            k = (str(z["stratum"][i]), int(z["percentile"][i]))
            worst = max(worst, float(np.abs(pred[i] - fig[k][cfg]).max()))
    print(f"  {key}: predictions of {', '.join(MODELS)} recomputed on the CPU, largest difference {worst:.1e} m")
    return worst < TOL_M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="unpacked results_fedtraj_20261025 (for the checkpoints)")
    ap.add_argument("--code", default=None, help="directory that contains the fedtraj package")
    args = ap.parse_args()
    report = json.loads((Q / "qualitative_r12.json").read_text(encoding="utf-8"))
    ok = True
    print("1. selection from the per-window errors")
    for key in SETTINGS:
        ok &= check_selection(key, report)
    print("2. figure data from the raw windows")
    figs = {key: figure_data(key) for key in SETTINGS}
    for key in SETTINGS:
        ok &= check_raw(key, figs[key])
    if args.results and args.code:
        print("3. predictions of the seed-one checkpoints")
        for key in SETTINGS:
            ok &= check_predictions(key, figs[key], args.results, args.code)
    else:
        print("3. skipped (pass --results and --code to re-run the checkpoints)")
    print("all checks passed" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
