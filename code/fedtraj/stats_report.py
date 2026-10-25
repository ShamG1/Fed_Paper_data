"""Model-free statistics of a fedtraj dataset: where the CV residual (what the CV-anchored model must learn)
is large, and how much it differs between clients.

Per partition and client (training windows):
    resid_rms      RMS of the CV residual over 50 steps x 2 coordinates (m); under local Adam the client's
                   update scale is normalized away, so the spread of this number is what SCA restores
    tail_share     share of the client's CV loss carried by its top-20% windows (DTS tail)
    bias_body/tail mean CV residual at 5 s (lat, lon) over the body / tail windows
Summary: max/min resid_rms across clients; between-client dispersion of the mean residual at 5 s relative
to its within-client RMS, separately for body and tail windows (a model-free proxy of how strongly the
first gradients of a CV-anchored model differ between clients in each stratum).

Usage
    python -m fedtraj.stats_report --data-dir data/fedtraj_ngsim [--out stats.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtraj import layout as L  # noqa: E402


def residuals(frames, starts, chunk=50000):
    loss, r5 = [], []
    for s in range(0, starts.shape[0], chunk):
        xy = L.gather(frames, starts[s:s + chunk], [0, 1]).astype(np.float64)
        true = xy[:, L.OBS:] - xy[:, L.OBS - 1:L.OBS]
        res = true - L.cv_displacement(xy)
        loss.append((res ** 2).mean(axis=(1, 2)))
        r5.append(res[:, -1])
    return np.concatenate(loss), np.concatenate(r5)


def summarize(loss, r5, lab, k, q=0.2):
    rows, means_b, means_t, n_b, n_t = [], [], [], [], []
    for c in range(k):
        m = lab == c
        lc, rc = loss[m], r5[m]
        thr = np.quantile(lc, 1 - q)
        tail = lc >= thr
        rows.append({"client": c, "windows": int(m.sum()), "resid_rms": float(np.sqrt(lc.mean())),
                     "tail_share": float(lc[tail].sum() / lc.sum()),
                     "bias_body_5s": rc[~tail].mean(0).round(3).tolist(),
                     "bias_tail_5s": rc[tail].mean(0).round(3).tolist()})
        means_b.append(rc[~tail].mean(0)); n_b.append((~tail).sum())
        means_t.append(rc[tail].mean(0)); n_t.append(tail.sum())
    rms = np.array([r["resid_rms"] for r in rows])

    def dispersion(means, counts, mask_fn):
        means, w = np.array(means), np.array(counts, float)
        w /= w.sum()
        mu = (w[:, None] * means).sum(0)
        between = np.sqrt((w[:, None] * (means - mu) ** 2).sum(0))
        within = []
        for c in range(k):
            m = lab == c
            lc = loss[m]
            sel = mask_fn(lc)
            within.append(np.sqrt((r5[m][sel] ** 2).mean(0)))
        within = (w[:, None] * np.array(within)).sum(0)
        return (between / within).round(3).tolist()

    disp_b = dispersion(means_b, n_b, lambda lc: lc < np.quantile(lc, 1 - q))
    disp_t = dispersion(means_t, n_t, lambda lc: lc >= np.quantile(lc, 1 - q))
    return {"clients": rows, "resid_rms_max_over_min": float(rms.max() / rms.min()),
            "tail_share_median": float(np.median([r["tail_share"] for r in rows])),
            "between_over_within_5s_body_latlon": disp_b, "between_over_within_5s_tail_latlon": disp_t}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.data_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    frames = np.load(d / "frames.npy", mmap_mode="r")
    tracks = np.load(d / "tracks.npz")
    st = np.load(d / "win_train.npy")
    owner = np.load(d / "wtrack_train.npy")
    loss, r5 = residuals(frames, st)
    report = {"dataset": d.name}
    for part, info in meta["partitions"].items():
        lab = tracks[part][owner].astype(np.int64)
        report[part] = summarize(loss, r5, lab, len(info["names"]))
        report[part]["names"] = info["names"]
        s = report[part]
        print(f"{d.name} / {part}: resid RMS per client "
              f"{[round(r['resid_rms'], 2) for r in s['clients']]}  (max/min {s['resid_rms_max_over_min']:.2f})")
        print(f"   tail (top 20%) share of CV loss, median {s['tail_share_median']:.1%}; "
              f"between/within dispersion of 5 s mean residual (lat, lon): body {s['between_over_within_5s_body_latlon']}"
              f"  tail {s['between_over_within_5s_tail_latlon']}")
    out = Path(args.out) if args.out else d / "stats_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
