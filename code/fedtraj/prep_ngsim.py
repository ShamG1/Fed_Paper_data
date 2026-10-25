"""NGSIM US-101 / I-80 raw trajectories (the six 15-minute periods) -> fedtraj dataset directory.

Reads the full ``trajectories-*.txt`` files inside the official zips (the ``.csv`` copies in the zips are
truncated at 1,048,575 rows).  Feet are converted to meters, duplicated (vehicle, frame) rows are dropped
and tracks are split at frame gaps.  Positions are NGSIM front-centre coordinates, unsmoothed; velocities
and accelerations are causal backward differences (``layout.causal_diff``).  Neighbours are searched per
frame on the lane grid: nearest vehicle ahead / behind in the same lane and in the two adjacent lanes.

Natural clients: the six (site, period) recordings.  Motion clients: K-means on observation-only
kinematics of training windows (``layout.motion_partition``).

Usage
    python -m fedtraj.prep_ngsim --out data/fedtraj_ngsim
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtraj import layout as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FT = 0.3048
SOURCES = [
    ("US101-0750", 0, "US-101-LosAngeles-CA/us-101-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.txt"),
    ("US101-0805", 0, "US-101-LosAngeles-CA/us-101-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0805am-0820am/trajectories-0805am-0820am.txt"),
    ("US101-0820", 0, "US-101-LosAngeles-CA/us-101-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0820am-0835am/trajectories-0820am-0835am.txt"),
    ("I80-1600", 1, "I-80-Emeryville-CA/i-80-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0400pm-0415pm/trajectories-0400-0415.txt"),
    ("I80-1700", 1, "I-80-Emeryville-CA/i-80-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0500pm-0515pm/trajectories-0500-0515.txt"),
    ("I80-1715", 1, "I-80-Emeryville-CA/i-80-vehicle-trajectory-data.zip",
     "vehicle-trajectory-data/0515pm-0530pm/trajectories-0515-0530.txt"),
]
# columns of the 18-column txt: 0 Vehicle_ID, 1 Frame_ID, 4 Local_X, 5 Local_Y, 10 v_Class, 13 Lane_ID
USECOLS = [0, 1, 4, 5, 10, 13]
NAMES = ["vid", "frame", "lx", "ly", "cls", "lane"]
RIGHTMOST_WITH_RIGHT_NEIGHBOUR = 5   # lanes 1..5 have a lane to their right on both sites


def neighbours(frame, lane, lon):
    """Row index of the nearest vehicle ahead / behind in the same, left (lane-1) and right (lane+1) lane."""
    n = frame.shape[0]
    g = frame.astype(np.int64) * 16 + lane.astype(np.int64)
    key = g.astype(np.float64) * 4096.0 + lon
    order = np.lexsort((lon, g))
    ks, gs = key[order], g[order]
    out = np.full((n, 6), -1, np.int64)
    for slot, dl in ((0, 0), (2, -1), (4, 1)):
        gq = g + dl
        kq = gq.astype(np.float64) * 4096.0 + lon
        ahead = np.searchsorted(ks, kq, side="right")
        behind = (np.searchsorted(ks, kq, side="left") if dl == 0 else ahead) - 1
        ok_a = (ahead < n) & (gs[np.minimum(ahead, n - 1)] == gq)
        ok_b = (behind >= 0) & (gs[np.maximum(behind, 0)] == gq)
        out[:, slot] = np.where(ok_a, order[np.minimum(ahead, n - 1)], -1)
        out[:, slot + 1] = np.where(ok_b, order[np.maximum(behind, 0)], -1)
    return out


def load_period(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        df = pd.read_csv(fh, sep=r"\s+", header=None, usecols=USECOLS, names=None, engine="c")
    df.columns = NAMES
    n0 = len(df)
    df = df.drop_duplicates(["vid", "frame"]).sort_values(["vid", "frame"]).reset_index(drop=True)
    return df, n0 - len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "1. NGSIM"))
    ap.add_argument("--out", default=str(ROOT / "data" / "fedtraj_ngsim"))
    ap.add_argument("--stride", type=int, default=15, help="window stride in frames (all splits)")
    ap.add_argument("--max-vehicles", type=int, default=0, help="debug: keep the first N vehicles per period")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats, tracks = [], {k: [] for k in ("offset", "length", "vehicle", "group", "location", "natural", "cls")}
    offset = 0
    report = []
    for gi, (name, site, zp, member) in enumerate(SOURCES):
        L.log(f"{name}: reading {member}")
        df, dup = load_period(Path(args.src) / zp, member)
        if args.max_vehicles:
            keep = np.unique(df["vid"].to_numpy())[:args.max_vehicles]
            df = df[df["vid"].isin(keep)].reset_index(drop=True)
        vid = df["vid"].to_numpy(np.int64)
        frame = df["frame"].to_numpy(np.int64)
        new = np.ones(len(df), bool)
        new[1:] = (vid[1:] != vid[:-1]) | (frame[1:] != frame[:-1] + 1)
        seg_start = np.flatnonzero(new)
        seg_len = np.diff(np.append(seg_start, len(df)))
        lat = df["lx"].to_numpy(np.float64) * FT
        lon = df["ly"].to_numpy(np.float64) * FT
        lane = df["lane"].to_numpy(np.int64)
        ego = {"lat": lat, "lon": lon}
        ego["v_lat"] = L.causal_diff(lat, seg_start, seg_len)
        ego["v_lon"] = L.causal_diff(lon, seg_start, seg_len)
        ego["a_lat"] = L.causal_diff(ego["v_lat"], seg_start, seg_len)
        ego["a_lon"] = L.causal_diff(ego["v_lon"], seg_start, seg_len)
        nb = neighbours(frame, lane, lon)
        flags = np.stack([(lane >= 2), (lane <= RIGHTMOST_WITH_RIGHT_NEIGHBOUR)], 1).astype(np.float32)
        feats.append(L.assemble_features(ego, nb, flags))
        cls = df["cls"].to_numpy(np.int64)
        tracks["offset"].append(seg_start + offset)
        tracks["length"].append(seg_len)
        tracks["vehicle"].append(gi * 100000 + vid[seg_start])
        tracks["group"].append(np.full(seg_start.shape[0], gi))
        tracks["location"].append(np.full(seg_start.shape[0], site))
        tracks["natural"].append(np.full(seg_start.shape[0], gi))
        tracks["cls"].append(cls[seg_start])
        offset += len(df)
        occupied = (nb >= 0).mean(0)
        report.append({"period": name, "rows": int(len(df)), "duplicates_dropped": int(dup),
                       "vehicles": int(np.unique(vid).shape[0]), "tracks": int(seg_start.shape[0]),
                       "slot_occupancy": [round(float(x), 3) for x in occupied]})
        L.log(f"  rows={len(df)} dup={dup} vehicles={report[-1]['vehicles']} tracks={seg_start.shape[0]} "
              f"slot occupancy={report[-1]['slot_occupancy']}")
        del df

    frames = np.concatenate(feats)
    del feats
    tracks = {k: np.concatenate(v) for k, v in tracks.items()}
    strides = {"train": args.stride, "valid": args.stride, "test": args.stride}
    extra = {"dataset": "NGSIM US-101 + I-80 (raw trajectories, 6 periods)", "periods": report,
             "positions": "front centre (NGSIM Local_X / Local_Y), meters, unsmoothed",
             "lane_flags": "left = lane>=2, right = lane<=5"}
    L.write_dataset(args.out, frames, tracks, strides, [s[0] for s in SOURCES], extra, seed=args.seed)
    L.log("done")


if __name__ == "__main__":
    main()
