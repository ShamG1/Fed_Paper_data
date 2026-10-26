"""GPU-resident windows, feature transform, CV-anchored prediction, physical loss and evaluation.

Input features of the 30 observed frames (all physical units, then standardized per channel with
training statistics):
    ego lat/lon relative to the last observed position; ego v_lat, v_lon, a_lat, a_lon;
    neighbours: rel_lat, rel_lon, velocity relative to the ego (v_nb - v_ego), a_lat, a_lon;
    left / right lane flags.
Prediction: displacement from the last observed position over the 50 future frames,
    pred = CV + res_scale * network_output,
where CV extrapolates the mean of the last ``CV_K`` observed increments and ``res_scale`` [50, 2] is the
training standard deviation of the CV residual (a fixed reparameterization; the objective is unchanged).
With ``anchor="none"`` (the LSTM backbone) there is no kinematic anchor:
    pred = y_mean + y_scale * network_output,
``y_mean`` and ``y_scale`` [50, 2] being the training mean and standard deviation of the displacement itself,
i.e. a fixed target standardization that is the same for every window.
Loss: mean squared displacement error in m^2 over the 50 steps and 2 coordinates, divided by the constant
``loss_scale`` (mean CV loss on the training split) for numerical conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

OBS, PRED, WIN = 30, 50, 80
HORIZON_STEPS = [9, 19, 29, 39, 49]          # 1..5 s inside the 50 future steps
HORIZON_NAMES = ["1s", "2s", "3s", "4s", "5s"]
LATERAL_MANEUVER_M = 3.0                      # |lateral displacement at 5 s| above this = lane-change window


class WindowStore:
    """All frames on the device; windows are gathered by start offset."""

    def __init__(self, data_dir, partition, device, max_windows=None, seed=0):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.device = device
        self.cv_k = int(self.meta["cv_k"])
        # copy in chunks so that several concurrent jobs do not each hold the whole array in host RAM
        fr = np.load(self.dir / "frames.npy", mmap_mode="r")
        self.frames = torch.empty(fr.shape, dtype=torch.float32, device=device)
        step = 1 << 20
        for s in range(0, fr.shape[0], step):
            self.frames[s:s + step] = torch.from_numpy(np.array(fr[s:s + step], dtype=np.float32))
        del fr
        tracks = np.load(self.dir / "tracks.npz")
        if partition not in self.meta["partitions"]:
            raise ValueError(f"unknown partition {partition}; have {list(self.meta['partitions'])}")
        self.partition = partition
        self.client_names = self.meta["partitions"][partition]["names"]
        self.num_clients = len(self.client_names)
        lab = tracks[partition].astype(np.int64)
        rng = np.random.default_rng(seed)
        self.starts, self.client = {}, {}
        for split in ("train", "valid", "test"):
            st = np.load(self.dir / f"win_{split}.npy")
            cl = lab[np.load(self.dir / f"wtrack_{split}.npy")]
            if max_windows:                     # debug: per-client cap, fixed subsample
                keep = []
                for c in range(self.num_clients):
                    idx = np.flatnonzero(cl == c)
                    if idx.size > max_windows:
                        idx = np.sort(rng.choice(idx, max_windows, replace=False))
                    keep.append(idx)
                keep = np.sort(np.concatenate(keep))
                st, cl = st[keep], cl[keep]
            self.starts[split] = torch.from_numpy(st).to(device)
            self.client[split] = torch.from_numpy(cl).to(device)
        self._ar = torch.arange(WIN, device=device)

    def windows(self, starts):
        return self.frames[starts[:, None] + self._ar[None, :]]            # [B, 80, 44]

    def client_starts(self, split, c):
        return self.starts[split][self.client[split] == c]

    def counts(self, split):
        return torch.bincount(self.client[split], minlength=self.num_clients).tolist()


class FeatureTransform:
    """Fixed input standardization and output scales, fitted on a fixed subsample of training windows.

    ``anchor``: "cv" (the predictor returns a correction of CV extrapolation) or "none" (it returns the
    standardized displacement).
    """

    ANCHORS = ("cv", "none")

    def __init__(self, store: WindowStore, n_fit=100000, seed=0, anchor="cv"):
        if anchor not in self.ANCHORS:
            raise ValueError(f"unknown anchor {anchor}; have {self.ANCHORS}")
        self.anchor = anchor
        self.store = store
        self.cv_k = store.cv_k
        dev = store.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        tr = store.starts["train"]
        sel = tr[torch.randperm(tr.numel(), generator=g)[:min(n_fit, tr.numel())].to(dev)]
        # running float64 sums instead of concatenating all fitting windows: keeps the peak GPU memory of a
        # job close to the size of frames.npy, so several jobs can share one GPU
        s1 = torch.zeros(44, device=dev, dtype=torch.float64)
        s2 = torch.zeros(44, device=dev, dtype=torch.float64)
        r1 = torch.zeros(PRED, 2, device=dev, dtype=torch.float64)
        r2 = torch.zeros(PRED, 2, device=dev, dtype=torch.float64)
        y1 = torch.zeros(PRED, 2, device=dev, dtype=torch.float64)      # displacement moments (anchor "none")
        y2 = torch.zeros(PRED, 2, device=dev, dtype=torch.float64)
        n_rows, n_win = 0, 0
        for s in range(0, sel.numel(), 10000):
            w = store.windows(sel[s:s + 10000])
            f = self._raw_inputs(w).reshape(-1, w.shape[-1]).double()
            s1 += f.sum(0)
            s2 += (f * f).sum(0)
            n_rows += f.shape[0]
            fut, cv = self._targets(w)
            r = (fut - cv).double()
            r1 += r.sum(0)
            r2 += (r * r).sum(0)
            y = fut.double()
            y1 += y.sum(0)
            y2 += (y * y).sum(0)
            n_win += r.shape[0]
            del w, f, r, y
        mean = s1 / n_rows
        self.mean = mean.float()
        self.std = (s2 / n_rows - mean * mean).clamp_min(0).sqrt().float().clamp_min(1e-3)
        rm = r1 / n_win
        self.res_scale = (r2 / n_win - rm * rm).clamp_min(0).sqrt().float().clamp_min(0.01)   # [50, 2]
        ym = y1 / n_win
        self.y_mean = ym.float()                                                              # [50, 2]
        self.y_scale = (y2 / n_win - ym * ym).clamp_min(0).sqrt().float().clamp_min(0.01)    # [50, 2]
        self.k = torch.arange(1, PRED + 1, device=dev, dtype=torch.float32)[None, :, None]
        self.loss_scale = float(store.meta["loss_scale"])

    @staticmethod
    def _raw_inputs(w):
        obs = w[:, :OBS].clone()
        last = obs[:, -1:, :]
        obs[:, :, 0:2] = obs[:, :, 0:2] - last[:, :, 0:2]
        for k in range(6):
            b = 6 + 6 * k
            obs[:, :, b + 2:b + 4] = obs[:, :, b + 2:b + 4] - obs[:, :, 2:4]
        return obs

    def _targets(self, w):
        last = w[:, OBS - 1, 0:2]
        fut = w[:, OBS:, 0:2] - last[:, None, :]
        inc = (last - w[:, OBS - 1 - self.cv_k, 0:2]) / self.cv_k
        cv = inc[:, None, :] * torch.arange(1, PRED + 1, device=w.device, dtype=w.dtype)[None, :, None]
        return fut, cv

    def batch(self, starts):
        """Standardized inputs [B, 30, 44], true future displacement [B, 50, 2], CV displacement [B, 50, 2]."""
        w = self.store.windows(starts)
        x = (self._raw_inputs(w) - self.mean) / self.std
        fut, cv = self._targets(w)
        return x, fut, cv

    def predict(self, model, x, cv):
        if self.anchor == "cv":
            return cv + model(x) * self.res_scale
        return self.y_mean + model(x) * self.y_scale

    def state(self):
        """Fitted statistics, stored with each checkpoint."""
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "res_scale": self.res_scale.cpu(),
                "y_mean": self.y_mean.cpu(), "y_scale": self.y_scale.cpu(), "anchor": self.anchor}


def window_sq_error(pred, fut):
    """Per-window mean squared displacement error over 50 steps x 2 coordinates (m^2)."""
    return ((pred - fut) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def score_windows(model, tf: FeatureTransform, starts, batch=4096):
    """Per-window physical loss under ``model`` in evaluation mode (used for sample scheduling)."""
    model.eval()
    out = torch.empty(starts.numel(), device=starts.device)
    for s in range(0, starts.numel(), batch):
        x, fut, cv = tf.batch(starts[s:s + batch])
        out[s:s + batch] = window_sq_error(tf.predict(model, x, cv), fut)
    return out


@torch.no_grad()
def evaluate(model, tf: FeatureTransform, split, batch=4096, keep_windows=False):
    """Coordinate RMSE at 1-5 s (mean over windows and the two coordinates), per-client 5 s RMSE, loss."""
    store = tf.store
    model.eval()
    starts, client = store.starts[split], store.client[split]
    n = starts.numel()
    sq_h = torch.zeros(5, device=starts.device, dtype=torch.float64)
    loss_sum = torch.zeros((), device=starts.device, dtype=torch.float64)
    cl_sq = torch.zeros(store.num_clients, device=starts.device, dtype=torch.float64)
    cl_n = torch.zeros(store.num_clients, device=starts.device, dtype=torch.float64)
    w5 = torch.empty(n, device=starts.device) if keep_windows else None
    cv5 = torch.empty(n, device=starts.device) if keep_windows else None
    lat5 = torch.empty(n, device=starts.device) if keep_windows else None
    for s in range(0, n, batch):
        x, fut, cv = tf.batch(starts[s:s + batch])
        pred = tf.predict(model, x, cv)
        err = pred - fut
        e2 = (err[:, HORIZON_STEPS] ** 2).sum(-1)                      # [B, 5] summed over 2 coords
        sq_h += e2.double().sum(0)
        loss_sum += ((err ** 2).mean(dim=(1, 2))).double().sum()
        c = client[s:s + batch]
        cl_sq.index_add_(0, c, e2[:, 4].double())
        cl_n.index_add_(0, c, torch.ones_like(c, dtype=torch.float64))
        if keep_windows:
            w5[s:s + batch] = e2[:, 4] / 2
            cv5[s:s + batch] = ((cv[:, 49] - fut[:, 49]) ** 2).sum(-1) / 2
            lat5[s:s + batch] = fut[:, 49, 0].abs()
    rmse = torch.sqrt(sq_h / (2 * n)).tolist()
    out = {h: float(v) for h, v in zip(HORIZON_NAMES, rmse)}
    out["mean"] = float(np.mean(rmse))
    out["loss_m2"] = float(loss_sum / n)
    out["per_client_5s"] = torch.sqrt(cl_sq / (2 * cl_n.clamp_min(1))).tolist()
    out["per_client_n"] = cl_n.long().tolist()
    if keep_windows:
        out["_w5"], out["_cv5"], out["_lat5"] = w5, cv5, lat5
    return out


def stratify(result):
    """5 s RMSE by CV-difficulty quintile (edges from the evaluated split) and for lane-change windows."""
    w5, cv5, lat5 = result["_w5"], result["_cv5"], result["_lat5"]
    edges = torch.quantile(cv5[torch.randperm(cv5.numel(), device=cv5.device)[:200000]],
                           torch.tensor([0.2, 0.4, 0.6, 0.8], device=cv5.device))
    q = torch.bucketize(cv5, edges)
    out = {f"Q{i + 1}": float(torch.sqrt(w5[q == i].mean())) for i in range(5)}
    lat = lat5 > LATERAL_MANEUVER_M
    out["lateral"] = float(torch.sqrt(w5[lat].mean())) if lat.any() else float("nan")
    out["lateral_share"] = float(lat.float().mean())
    out["cv_Q"] = {f"Q{i + 1}": float(torch.sqrt(cv5[q == i].mean())) for i in range(5)}
    return out
