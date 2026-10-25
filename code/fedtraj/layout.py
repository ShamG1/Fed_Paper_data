"""Shared preprocessing for NGSIM and highD.

Frame layout (44 channels, all in meters / seconds, expressed in the ego vehicle's driving frame:
lat increases to the right of the driving direction, lon increases along it)

    0..5    ego: lat, lon, v_lat, v_lon, a_lat, a_lon
    6..41   six neighbour slots x [rel_lat, rel_lon, v_lat, v_lon, a_lat, a_lon]
            slot order: same-lane ahead, same-lane behind, left ahead, left behind, right ahead, right behind
            rel_* = neighbour minus ego; velocities/accelerations are the neighbour's own.
            Empty slot (or |rel_lon| > NEIGHBOUR_RANGE): rel_lat = nominal lane offset, rel_lon = +/-RANGE,
            velocities = ego velocities (zero relative speed), accelerations = 0.
    42, 43  left lane exists, right lane exists (0/1)

Velocities and accelerations are causal backward differences over KIN_LAG frames of the 10 Hz positions,
so no feature of an observed frame depends on a later frame.  Positions are not smoothed.

Dataset directory written by ``write_dataset``
    frames.npy           float32 [F, 44], all tracks concatenated
    tracks.npz           per track: offset, length, vehicle, group, location, split, natural, motion, cls
    win_{split}.npy      int64 window start offsets into frames.npy (window = 80 consecutive frames)
    wtrack_{split}.npy   int32 track index of every window
    meta.json            channel names, clients of each partition, range floor, loss scale, CV reference
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FPS = 10
DT = 1.0 / FPS
OBS = 30            # observed frames (3 s)
PRED = 50           # predicted frames (5 s)
WIN = OBS + PRED    # 80
KIN_LAG = 5         # backward-difference lag (0.5 s) for velocity and acceleration
START_MIN = 2 * KIN_LAG   # first frame of a track that may start a window (kinematics fully defined)
CV_K = 5            # CV reference / anchor: mean of the last CV_K observed increments
NEIGHBOUR_RANGE = 60.0
LANE_OFFSET = 3.7
HORIZONS = {f"{h}s": OBS - 1 + FPS * h for h in range(1, 6)}   # frame index inside the window

SLOTS = ["same_ahead", "same_behind", "left_ahead", "left_behind", "right_ahead", "right_behind"]
SENT_LAT = np.array([0.0, 0.0, -LANE_OFFSET, -LANE_OFFSET, LANE_OFFSET, LANE_OFFSET], np.float32)
SENT_LON = np.array([1, -1, 1, -1, 1, -1], np.float32) * NEIGHBOUR_RANGE
CHANNELS = (["lat", "lon", "v_lat", "v_lon", "a_lat", "a_lon"]
            + [f"{s}_{q}" for s in SLOTS for q in ("rel_lat", "rel_lon", "v_lat", "v_lon", "a_lat", "a_lon")]
            + ["left_lane", "right_lane"])
assert len(CHANNELS) == 44


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- kinematics
def causal_diff(x, starts, lengths, lag=KIN_LAG, dt=DT):
    """Backward difference (x[t] - x[t-l]) / (l*dt) with l = min(t, lag) inside every track.

    ``x`` holds all tracks concatenated; ``starts``/``lengths`` delimit the tracks.  The first frame of a
    track copies the second one.
    """
    out = np.zeros_like(x, dtype=np.float64)
    for s, n in zip(starts, lengths):
        seg = x[s:s + n].astype(np.float64)
        if n < 2:
            continue
        t = np.arange(n)
        back = np.minimum(t, lag)
        d = np.empty(n)
        d[1:] = (seg[1:] - seg[t[1:] - back[1:]]) / (back[1:] * dt)
        d[0] = d[1]
        out[s:s + n] = d
    return out


def assemble_features(ego, nb, flags):
    """Build the [N, 44] frame features.

    ego   dict of (N,) arrays: lat, lon, v_lat, v_lon, a_lat, a_lon
    nb    (N, 6) int64 row index of the neighbour in each slot (into the same arrays), -1 if none
    flags (N, 2) left / right lane exists
    """
    n = ego["lat"].shape[0]
    f = np.empty((n, 44), np.float32)
    keys = ["lat", "lon", "v_lat", "v_lon", "a_lat", "a_lon"]
    for c, k in enumerate(keys):
        f[:, c] = ego[k]
    for k in range(6):
        j = nb[:, k]
        safe = np.where(j >= 0, j, 0)
        rel_lat = ego["lat"][safe] - ego["lat"]
        rel_lon = ego["lon"][safe] - ego["lon"]
        has = (j >= 0) & (np.abs(rel_lon) <= NEIGHBOUR_RANGE)
        b = 6 + 6 * k
        f[:, b] = np.where(has, rel_lat, SENT_LAT[k])
        f[:, b + 1] = np.where(has, rel_lon, SENT_LON[k])
        f[:, b + 2] = np.where(has, ego["v_lat"][safe], ego["v_lat"])
        f[:, b + 3] = np.where(has, ego["v_lon"][safe], ego["v_lon"])
        f[:, b + 4] = np.where(has, ego["a_lat"][safe], 0.0)
        f[:, b + 5] = np.where(has, ego["a_lon"][safe], 0.0)
    f[:, 42:44] = flags
    return f


# ----------------------------------------------------------------------------- windows and splits
def split_tracks(vehicle, group, fracs=(0.8, 0.1, 0.1), seed=42):
    """Vehicle-level split, stratified by ``group``: every track of a vehicle lands in one split."""
    rng = np.random.default_rng(seed)
    split = np.full(vehicle.shape[0], -1, np.int8)
    for g in np.unique(group):
        m = group == g
        vids = np.unique(vehicle[m])
        rng.shuffle(vids)
        n_tr = int(round(fracs[0] * len(vids)))
        n_va = int(round(fracs[1] * len(vids)))
        lab = {v: 0 for v in vids[:n_tr]}
        lab.update({v: 1 for v in vids[n_tr:n_tr + n_va]})
        lab.update({v: 2 for v in vids[n_tr + n_va:]})
        idx = np.flatnonzero(m)
        split[idx] = [lab[v] for v in vehicle[idx]]
    assert (split >= 0).all()
    return split


def window_starts(offsets, lengths, stride):
    """Window start offsets (into the frame array) and the owning track of each window."""
    starts, owner = [], []
    for t, (o, n) in enumerate(zip(offsets, lengths)):
        if n < START_MIN + WIN:
            continue
        s = np.arange(START_MIN, n - WIN + 1, stride, dtype=np.int64)
        starts.append(o + s)
        owner.append(np.full(s.shape[0], t, np.int32))
    if not starts:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    return np.concatenate(starts), np.concatenate(owner)


def gather(frames, starts, cols=None):
    idx = starts[:, None] + np.arange(WIN)[None, :]
    w = frames[idx]
    return w if cols is None else w[:, :, cols]


# ----------------------------------------------------------------------------- statistics
def cv_displacement(xy):
    """CV reference on [N, 80, 2] positions: future displacement from frame OBS-1, shape [N, PRED, 2]."""
    inc = (xy[:, OBS - 1] - xy[:, OBS - 1 - CV_K]) / CV_K
    k = np.arange(1, PRED + 1, dtype=np.float64)[None, :, None]
    return inc[:, None, :] * k


def cv_reference(frames, starts, chunk=50000):
    """Coordinate RMSE of CV at 1-5 s and the mean CV loss (mean over 50 steps x 2 coords, m^2)."""
    sq_h = {k: 0.0 for k in HORIZONS}
    loss_sum, n = 0.0, 0
    for s in range(0, starts.shape[0], chunk):
        xy = gather(frames, starts[s:s + chunk], [0, 1]).astype(np.float64)
        true = xy[:, OBS:] - xy[:, OBS - 1:OBS]
        err = cv_displacement(xy) - true
        loss_sum += float((err ** 2).mean(axis=(1, 2)).sum())
        for k, fi in HORIZONS.items():
            sq_h[k] += float((err[:, fi - (OBS - 1) - 1] ** 2).sum())
        n += xy.shape[0]
    rmse = {k: float(np.sqrt(v / (2 * n))) for k, v in sq_h.items()}
    rmse["mean"] = float(np.mean(list(rmse.values())))
    return rmse, loss_sum / max(n, 1)


def range_floor(frames, starts, q=5.0, max_windows=200000, seed=0):
    """Per-channel q-th percentile of the observed (first OBS frames) min-max range on training windows."""
    rng = np.random.default_rng(seed)
    sel = starts if starts.shape[0] <= max_windows else rng.choice(starts, max_windows, replace=False)
    out = np.empty((sel.shape[0], 44), np.float32)
    for s in range(0, sel.shape[0], 50000):
        obs = gather(frames, sel[s:s + 50000])[:, :OBS]
        out[s:s + obs.shape[0]] = obs.max(1) - obs.min(1)
    return np.percentile(out, q, axis=0).astype(np.float32)


def motion_features(frames, starts):
    """Observation-only kinematic summary of each window (used only to build the motion partition)."""
    feats = []
    for s in range(0, starts.shape[0], 50000):
        w = gather(frames, starts[s:s + 50000])[:, :OBS].astype(np.float64)
        gap = np.clip(w[:, :, 7], 0, NEIGHBOUR_RANGE)                   # same-lane ahead rel_lon
        present = (np.abs(w[:, :, 7:42:6]) < NEIGHBOUR_RANGE).mean(axis=(1, 2))
        feats.append(np.stack([
            w[:, :, 3].mean(1), w[:, :, 3].std(1),                       # v_lon mean / std
            np.abs(w[:, :, 2]).mean(1),                                  # |v_lat|
            w[:, :, 5].mean(1), w[:, :, 5].std(1),                       # a_lon mean / std
            np.abs(w[:, -1, 0] - w[:, 0, 0]),                            # lateral displacement
            gap.mean(1),                                                 # gap to leader
            (w[:, :, 9] - w[:, :, 3]).mean(1),                           # leader relative speed
            present,
        ], axis=1))
    return np.concatenate(feats).astype(np.float32)


def motion_partition(frames, tracks, train_starts, train_owner, k=10, seed=42, min_share=0.02, trials=200):
    """K-means on per-track mean kinematic features of training windows; val/test tracks -> nearest centroid.

    Features are winsorized at the 1st/99th percentile and standardized.  Among ``trials`` single-start
    K-means fits, the lowest-inertia solution whose every client holds at least ``min_share`` of the
    training windows is kept, so that no client is a handful of outlier tracks.  If no fit satisfies the
    constraint, the fit with the largest smallest-client share is kept and a warning is logged.
    """
    from sklearn.cluster import KMeans

    n_tracks = tracks["offset"].shape[0]
    f_win = motion_features(frames, train_starts)
    dim = f_win.shape[1]
    acc = np.zeros((n_tracks, dim))
    cnt = np.zeros(n_tracks)
    np.add.at(acc, train_owner, f_win)
    np.add.at(cnt, train_owner, 1)
    tr = cnt > 0
    mu = acc[tr] / cnt[tr, None]
    lo, hi = np.percentile(mu, 1, axis=0), np.percentile(mu, 99, axis=0)
    mean, std = np.clip(mu, lo, hi).mean(0), np.clip(mu, lo, hi).std(0) + 1e-9

    def z(x):
        return (np.clip(x, lo, hi) - mean) / std

    zt = z(mu)
    w = cnt[tr]
    best, fallback, fallback_share = None, None, -1.0
    for s in range(trials):
        km = KMeans(n_clusters=k, n_init=1, random_state=seed + s).fit(zt)
        share = np.bincount(km.labels_, weights=w, minlength=k) / w.sum()
        if share.min() >= min_share and (best is None or km.inertia_ < best.inertia_):
            best = km
        if share.min() > fallback_share:
            fallback, fallback_share = km, float(share.min())
    if best is None:
        log(f"  WARNING: no K-means fit has every client >= {min_share:.1%} of the training windows; "
            f"keeping the fit whose smallest client holds {fallback_share:.2%}")
        best = fallback
    lab = np.full(n_tracks, -1, np.int16)
    lab[np.flatnonzero(tr)] = best.labels_
    # val/test tracks: nearest centroid of their own observation-only features
    all_starts, all_owner = window_starts(tracks["offset"], tracks["length"], stride=WIN)
    f_all = motion_features(frames, all_starts)
    acc2 = np.zeros((n_tracks, dim))
    cnt2 = np.zeros(n_tracks)
    np.add.at(acc2, all_owner, f_all)
    np.add.at(cnt2, all_owner, 1)
    rest = (~tr) & (cnt2 > 0)
    if rest.any():
        lab[np.flatnonzero(rest)] = best.predict(z(acc2[rest] / cnt2[rest, None]))
    lab[lab < 0] = 0          # tracks too short for any window never matter
    centers = best.cluster_centers_ * std + mean
    return lab, centers.tolist()


# ----------------------------------------------------------------------------- writer
def partition_meta(frames, wins, labels, names):
    """Window counts per split and mean observed v_lon of the training windows for one track-level partition."""
    counts = {split: np.bincount(labels[ow], minlength=len(names)).tolist() for split, (st, ow) in wins.items()}
    st, ow = wins["train"]
    lab = labels[ow]
    v = np.concatenate([gather(frames, st[s:s + 50000], [3])[:, :OBS, 0].mean(1)
                        for s in range(0, st.shape[0], 50000)])
    speed = [float(v[lab == c].mean()) if (lab == c).any() else float("nan") for c in range(len(names))]
    return {"names": names, "counts": counts, "mean_v_lon": speed}


def write_dataset(out_dir, frames, tracks, strides, natural_names, extra_meta, motion_k=10, seed=42,
                  extra_partitions=None):
    """Write a dataset directory.  ``tracks`` must contain offset, length, vehicle, group, location, natural, cls.

    ``extra_partitions``: optional {name: (track-level labels, client names)} written next to natural / motion.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks = dict(tracks)
    tracks["split"] = split_tracks(tracks["vehicle"], tracks["group"], seed=seed)
    wins = {}
    for sp, name in enumerate(("train", "valid", "test")):
        m = tracks["split"] == sp
        off = np.where(m, tracks["offset"], 0)
        ln = np.where(m, tracks["length"], 0)                 # length 0 -> no windows for other splits
        st, ow = window_starts(off, ln, strides[name])
        wins[name] = (st, ow)
        log(f"  {name}: tracks={int(m.sum())}  windows={st.shape[0]}  (stride {strides[name]})")
    log("  motion partition (K-means on observation-only kinematics of training windows)")
    tracks["motion"], centers = motion_partition(frames, tracks, *wins["train"], k=motion_k, seed=seed)
    floor = range_floor(frames, wins["train"][0])
    cv_tr, loss_scale = cv_reference(frames, wins["train"][0])
    cv_va, _ = cv_reference(frames, wins["valid"][0])
    cv_te, _ = cv_reference(frames, wins["test"][0])

    log(f"  writing {out}")
    np.save(out / "frames.npy", frames.astype(np.float32))
    for name, (st, ow) in wins.items():
        np.save(out / f"win_{name}.npy", st)
        np.save(out / f"wtrack_{name}.npy", ow)

    partitions = {
        "natural": partition_meta(frames, wins, tracks["natural"].astype(np.int64), natural_names),
        "motion": {**partition_meta(frames, wins, tracks["motion"].astype(np.int64),
                                    [f"M{c}" for c in range(motion_k)]), "centers": centers},
    }
    for pname, (labels, names) in (extra_partitions or {}).items():
        tracks[pname] = np.asarray(labels, np.int64)
        partitions[pname] = partition_meta(frames, wins, tracks[pname], names)
    np.savez(out / "tracks.npz", **{k: np.asarray(v) for k, v in tracks.items()})

    meta = {
        "channels": CHANNELS, "fps": FPS, "obs": OBS, "pred": PRED, "kin_lag": KIN_LAG, "cv_k": CV_K,
        "start_min": START_MIN, "strides": strides, "neighbour_range": NEIGHBOUR_RANGE,
        "n_frames": int(frames.shape[0]), "n_tracks": int(tracks["offset"].shape[0]),
        "partitions": partitions,
        "range_floor": floor.tolist(),
        "loss_scale": loss_scale,
        "cv_rmse": {"train": cv_tr, "valid": cv_va, "test": cv_te},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta.update(extra_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"  CV test RMSE {cv_te}  loss scale {loss_scale:.3f} m^2")
    return meta
