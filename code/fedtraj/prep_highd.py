"""highD recordings (60) -> fedtraj dataset directory.

Every track is resampled from 25 Hz to the 10 Hz grid t_k = k / 10 s by linear interpolation of the
bounding-box centre; lane id and neighbour ids are taken from the nearest original frame.  Coordinates are
rotated into the driving frame (lon along the direction of travel, lat to its right), so both driving
directions share one convention.  Velocities and accelerations are recomputed from the resampled positions
with the same causal backward differences as NGSIM; the dataset's own (smoothed, non-causal) velocities are
not used.  Neighbour slots use highD's preceding / following ids; for the adjacent lanes the "alongside"
vehicle fills the ahead or behind slot when that slot is empty, according to the sign of its offset.

Natural clients: the six recording locations.  Motion clients: K-means on observation-only kinematics.

Usage
    python -m fedtraj.prep_highd --src highd-dataset-v1.0/data --out data/fedtraj_highd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtraj import layout as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NB_COLS = ["precedingId", "followingId", "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
           "rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
USE = ["frame", "id", "x", "y", "width", "height", "laneId"] + NB_COLS


def resample(tr, fps):
    """Interpolate every vehicle to the 10 Hz grid; returns a DataFrame indexed by (id, k)."""
    out = []
    for vid, g in tr.groupby("id", sort=True):
        t = (g["frame"].to_numpy() - 1) / fps
        k0, k1 = int(np.ceil(t[0] * L.FPS - 1e-9)), int(np.floor(t[-1] * L.FPS + 1e-9))
        if k1 < k0:
            continue
        k = np.arange(k0, k1 + 1)
        tk = k / L.FPS
        near = np.clip(np.floor(tk * fps + 0.5).astype(np.int64) - (g["frame"].iloc[0] - 1), 0, len(g) - 1)
        cx = g["x"].to_numpy() + g["width"].to_numpy() / 2
        cy = g["y"].to_numpy() + g["height"].to_numpy() / 2
        d = {"id": np.full(k.shape[0], vid), "k": k,
             "cx": np.interp(tk, t, cx), "cy": np.interp(tk, t, cy),
             "laneId": g["laneId"].to_numpy()[near]}
        for c in NB_COLS:
            d[c] = g[c].to_numpy()[near]
        out.append(pd.DataFrame(d))
    return pd.concat(out, ignore_index=True)


def lane_flags(lane, direction):
    """Left / right lane exists in the direction of travel (lane ids grow downwards in the image)."""
    left = np.zeros(lane.shape[0], bool)
    right = np.zeros(lane.shape[0], bool)
    for d in (1, 2):
        m = direction == d
        ids = set(np.unique(lane[m]).tolist())
        lo = np.array([x - 1 in ids for x in lane[m]])
        hi = np.array([x + 1 in ids for x in lane[m]])
        # direction 2 travels towards +x: its right side is +y (larger lane id); direction 1 is mirrored
        left[m], right[m] = (lo, hi) if d == 2 else (hi, lo)
    return np.stack([left, right], 1).astype(np.float32)


def process_recording(path_prefix):
    rec = pd.read_csv(f"{path_prefix}_recordingMeta.csv").iloc[0]
    tmeta = pd.read_csv(f"{path_prefix}_tracksMeta.csv").set_index("id")
    tr = pd.read_csv(f"{path_prefix}_tracks.csv", usecols=USE)
    df = resample(tr, float(rec["frameRate"]))
    df = df.sort_values(["id", "k"]).reset_index(drop=True)
    direction = tmeta.loc[df["id"].to_numpy(), "drivingDirection"].to_numpy()
    sign = np.where(direction == 1, -1.0, 1.0)
    lon = sign * df["cx"].to_numpy()
    lat = sign * df["cy"].to_numpy()

    vid = df["id"].to_numpy(np.int64)
    k = df["k"].to_numpy(np.int64)
    new = np.ones(len(df), bool)
    new[1:] = (vid[1:] != vid[:-1]) | (k[1:] != k[:-1] + 1)
    seg_start = np.flatnonzero(new)
    seg_len = np.diff(np.append(seg_start, len(df)))
    ego = {"lat": lat, "lon": lon}
    ego["v_lat"] = L.causal_diff(lat, seg_start, seg_len)
    ego["v_lon"] = L.causal_diff(lon, seg_start, seg_len)
    ego["a_lat"] = L.causal_diff(ego["v_lat"], seg_start, seg_len)
    ego["a_lon"] = L.causal_diff(ego["v_lon"], seg_start, seg_len)

    # neighbour rows on the same grid step
    key = vid * 10_000_000 + k
    order = np.argsort(key)
    ks = key[order]

    def row_of(ids):
        q = ids.astype(np.int64) * 10_000_000 + k
        pos = np.clip(np.searchsorted(ks, q), 0, len(ks) - 1)
        return np.where((ids > 0) & (ks[pos] == q), order[pos], -1)

    r = {c: row_of(df[c].to_numpy()) for c in NB_COLS}

    def rel_lon(rows):
        return np.where(rows >= 0, lon[np.maximum(rows, 0)] - lon, 0.0)

    nb = np.full((len(df), 6), -1, np.int64)
    nb[:, 0] = r["precedingId"]
    nb[:, 1] = r["followingId"]
    for slot, side in ((2, "left"), (4, "right")):
        ahead, along, behind = r[f"{side}PrecedingId"], r[f"{side}AlongsideId"], r[f"{side}FollowingId"]
        al = rel_lon(along)
        nb[:, slot] = np.where(ahead >= 0, ahead, np.where((along >= 0) & (al >= 0), along, -1))
        nb[:, slot + 1] = np.where(behind >= 0, behind, np.where((along >= 0) & (al < 0), along, -1))
    flags = lane_flags(df["laneId"].to_numpy(), direction)
    feats = L.assemble_features(ego, nb, flags)
    cls = (tmeta.loc[vid[seg_start], "class"].to_numpy() == "Truck").astype(np.int64)
    info = {"location": int(rec["locationId"]), "rows": int(len(df)), "vehicles": int(np.unique(vid).shape[0]),
            "slot_occupancy": [round(float(x), 3) for x in (nb >= 0).mean(0)]}
    return feats, seg_start, seg_len, vid[seg_start], cls, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "highd-dataset-v1.0" / "data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "fedtraj_highd"))
    ap.add_argument("--stride", type=int, default=10, help="window stride in frames (all splits)")
    ap.add_argument("--recordings", type=int, nargs="*", default=None, help="debug: subset of recording ids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = args.recordings or list(range(1, 61))
    feats, tracks = [], {k: [] for k in ("offset", "length", "vehicle", "group", "location", "natural", "cls")}
    report, offset = [], 0
    for rid in ids:
        f, s, n, v, cls, info = process_recording(str(Path(args.src) / f"{rid:02d}"))
        feats.append(f)
        tracks["offset"].append(s + offset)
        tracks["length"].append(n)
        tracks["vehicle"].append(rid * 100000 + v)
        tracks["group"].append(np.full(s.shape[0], rid))
        tracks["location"].append(np.full(s.shape[0], info["location"]))
        tracks["natural"].append(np.full(s.shape[0], info["location"] - 1))
        tracks["cls"].append(cls)
        offset += f.shape[0]
        info["recording"] = rid
        report.append(info)
        L.log(f"recording {rid:02d}: location {info['location']} rows={info['rows']} "
              f"vehicles={info['vehicles']} occupancy={info['slot_occupancy']}")
    frames = np.concatenate(feats)
    del feats
    tracks = {k: np.concatenate(v) for k, v in tracks.items()}
    locs = sorted(set(int(x) for x in tracks["location"]))
    if locs != list(range(1, len(locs) + 1)):          # debug subsets: re-index locations densely
        remap = {l: i for i, l in enumerate(locs)}
        tracks["natural"] = np.array([remap[int(x)] for x in tracks["location"]])
    names = [f"highD-L{l}" for l in locs]
    strides = {"train": args.stride, "valid": args.stride, "test": args.stride}
    extra = {"dataset": "highD v1.0 (60 recordings, resampled to 10 Hz)", "recordings": report,
             "positions": "bounding-box centre in the driving frame, meters, linear interpolation 25->10 Hz",
             "lane_flags": "adjacent lane id observed for the same driving direction in the recording"}
    # fine-grained natural partition: one client per recording (drone flight), for large-federation runs
    rec_ids = sorted(set(int(x) for x in tracks["group"]))
    rec_index = {r: i for i, r in enumerate(rec_ids)}
    recording = np.array([rec_index[int(x)] for x in tracks["group"]], np.int64)
    L.write_dataset(args.out, frames, tracks, strides, names, extra, seed=args.seed,
                    extra_partitions={"recording": (recording, [f"R{r:02d}" for r in rec_ids])})
    L.log("done")


if __name__ == "__main__":
    main()
