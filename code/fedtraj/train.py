"""Centralized and federated training on a fedtraj dataset directory.

Algorithms (``--algo``)
    central    pooled training (reference)
    fedavg     FedAvg with local AdamW (``--server-lr`` gives FedAvg with a server learning rate)
    fedprox    FedAvg + proximal term ``--mu``
    fednova    normalized averaging by local step counts
    fedavgm    server momentum (``--server-momentum``, default server lr 0.1)
    fedadam    server Adam (``--server-lr`` default 0.01, betas 0.9 / 0.99, eps 1e-3)
    scaffold   control variates, option I: c_i = gradient of the local objective at the broadcast model,
               estimated on ``--scaffold-batches`` batches; corrections are added to the gradients before AdamW
    fedbss     loss-ordered progressive admission (regression adaptation of FedBSS, cosine pacing)
    cqs        curriculum-quota scheduling, the client layer of Fed-SSBR: every round each client ranks its windows
               by the loss of the broadcast model (evaluation mode) and, after ``--cqs-warmup`` full-data rounds,
               local epoch e admits the lowest-loss fraction q_e, rising from ``--cqs-floor`` to 1 (linear by
               default; 0.6 / 0.8 / 1.0 for E = 3, i.e. visit counts 3 : 2 : 1 for the lowest 60 %, the next 20 %
               and the top 20 %).  The admitted windows are shuffled within each epoch.  ``--cqs-order random``
               (a fresh random order per client and round) and ``anti`` (reversed loss order) apply exactly the
               same per-epoch counts: budget-matched controls of the admission order.
    fedlc      local + global correction for local adaptive optimizers (Wang et al., 2021): the update of client
               i is divided by its accumulated diagonal preconditioner N_i = lr * sum_k 1/(sqrt(vhat_k)+eps),
               and the server rescales with N_s = (sum_i p_i / N_i)^-1, i.e. per-coordinate weights
               p_i N_s / N_i on the local updates
    fedtr      proposed: objective-invariant scheduling at two levels.  Sample level: tail routing with weight
               compensation.  Client level: a per-client routing budget q_i set by the server from the
               client's drift (``--budget adaptive``), made consistent under AdamW by the fedlc aggregation
               (``--agg lc``; ``--agg avg`` = FedAvg weights).  ``--budget fixed`` = the same q for every client.

FedTR, client i, round t (broadcast model theta^t, E local epochs, tail fraction q):
    score all windows with theta^t (physical loss, evaluation mode); tail T_i = top q of the client's windows
    (``--route random``: a random subset of the same size), body B_i = the rest.
    ``--tail-pos early`` (default): epoch 1 trains on all windows, epochs 2..E on the body only, so every tail
    window is visited once, at the start of the round, while the local model is still close to theta^t.
    ``late``: the mirror image (tail only in the last epoch, as in curriculum-style schedules).
    ``--tail-weight preserve`` (default): the single tail visit carries loss weight E, the weight FedAvg spreads
    over E visits, so with plain SGD every window keeps exactly its FedAvg weight per round (the first-order
    round objective is unchanged; only when the tail is seen changes).  ``--tail-weight one``: weight 1, i.e. the
    tail simply gets fewer visits, which down-weights it as curriculum-style schedules do.
    Training visits per round: (E (1-q) + q) n_i = 2.6 n_i for E = 3, q = 0.2 (FedBSS: 2.6 n_i; FedAvg: 3 n_i).

FedTR client level (``--budget adaptive``): because of the weight compensation, any routing budget q_i leaves the
    first-order round objective of client i unchanged under SGD, so the server may choose q_i per client without
    creating objective inconsistency.  It allocates the local visits s_i = 1 + (E-1)(1-q_i) per window so as to
    minimize the second-order drift term sum_i p_i s_i^2 zeta_i^2 at the fixed mean visit budget of the uniform
    schedule; the minimizer is s_i proportional to 1 / zeta_i^2 (``Server.budgets``).  zeta_i^2 is estimated by
    d_i = 1 - EMA(cos_i), cos_i being the cosine between the client's update and the sample-weighted mean update
    of the other clients (leave-one-out, from the uploaded models only).  A drifting client repeats fewer of its
    windows; an aligned client keeps more local steps.  q_i is bounded to [--q-min, --q-max].  Under AdamW the
    residual inconsistency of unequal local work is removed by the aggregation rule of Wang et al. (2021), local +
    global correction (``--agg lc``), used here as a component; ``fedlc`` is that rule without routing.  No extra
    upload beyond the fedlc preconditioner.

Server option ``--reweight udr`` (update-deviation reweighting, the server layer of Fed-SSBR; with the FedAvg-type
    aggregation of fedavg, fedprox, fedbss and cqs): over the trainable parameters of the participating clients,
    d_i = ||Delta_i - mean_j Delta_j|| (unweighted cohort mean), r_i = exp(-d_i / (median_j d_j * tau_d)),
    s_i = clip(r_i / sum_j p_j r_j, s_min, s_max) and alpha_i = p_i s_i / sum_j p_j s_j, p_i being the sample share
    within the cohort.  Parameters and floating buffers are averaged with alpha.  Every weight stays within
    [s_min / s_max, s_max / s_min] times its sample share, and equal deviations give FedAvg exactly.
    Fed-SSBR = ``--algo cqs --reweight udr``; the server layer alone = ``--algo fedavg --reweight udr``.
    ``--reweight udr-fs``: the full-state variant, whose deviations also include the floating buffers (BatchNorm
    running statistics).  ``--reweight static --static-beta b``: static weights alpha_i proportional to n_i^b within
    the cohort, the reference with the same weight concentration as UDR (``jobs.py --ssbr-select`` matches b to
    the mean effective number of clients of Fed-SSBR in the pilot).
FedBSS-mean (``--algo fedbss --bss-split mean``): the always-admitted low-loss group of each round is the set of
    windows whose loss does not exceed the client's mean loss, instead of the fixed fraction ``--bss-min``.
Checks added for the revision after R16 (2026-10-26, after all results of the paper):
    ``--cqs-direction descending``: the CQS quotas in reverse order, 1.0 / 0.8 / 0.6 for E = 3, with the same
    ranking (computed once per round).  Every window keeps its visit count (3 : 2 : 1); the highest-loss fifth is
    used in the first local epoch instead of the last, and the last epoch uses the lowest-loss 60 % only.
    ``--bn-recal all``: after the local epochs, re-estimate the BatchNorm running statistics of the local model on
    all windows of the client (one pass in training mode without gradients, cumulative average).  With PyTorch's
    momentum 0.1 the uploaded statistics otherwise come almost entirely from the last local epoch, which under
    descending quotas holds the lowest-loss windows only.
    ``--backbone lstm``: the LSTM encoder-decoder of ``model.py`` instead of the CV-anchored predictor; it
    predicts the standardized displacement without the CV anchor (``data.FeatureTransform``, anchor "none").
Combinations with existing rules (added 2026-10-24, after the main results):
    ``--nova``: aggregate with the FedNova rule of ``--algo fednova`` (each update divided by the client's actual
    number of local optimizer steps, rescaled by tau_eff = sum_i p_i tau_i) instead of FedAvg; with fedprox, fedbss
    and cqs.  ``--prox-mu MU``: add the FedProx proximal term (MU / 2) ||theta - theta^t||^2 of ``--algo fedprox`` to
    local training; with fedbss and cqs.  CQS+FedNova = ``--algo cqs --nova``; CQS+FedProx = ``--algo cqs --prox-mu
    MU``.  During the CQS warm-up rounds both reproduce ``--algo fednova`` / ``--algo fedprox --mu MU`` exactly.

Outputs (``--out-dir``/<dataset>/<partition>/<run_id>/): log.csv (one row per round / epoch), result.json,
best.pt (validation-best global model).

Example
    python -m fedtraj.train --data-dir data/fedtraj_ngsim --partition natural --algo fedtr --seed 1
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # only used with --deterministic 1

import argparse                     # noqa: E402
import csv                          # noqa: E402
import json                         # noqa: E402
import math                         # noqa: E402
import sys                          # noqa: E402
import time                         # noqa: E402
from pathlib import Path            # noqa: E402

import numpy as np                  # noqa: E402
import torch                        # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedtraj.data import (FeatureTransform, WindowStore, evaluate, score_windows,  # noqa: E402
                          stratify, window_sq_error)
from fedtraj.model import BACKBONES, build_model, count_parameters  # noqa: E402

ALGOS = ["central", "fedavg", "fedprox", "fednova", "fedavgm", "fedadam", "scaffold", "fedbss", "cqs", "fedlc",
         "fedtr"]
UDR_ALGOS = ("fedavg", "fedprox", "fedbss", "cqs")    # FedAvg-type aggregation, so the weights can be rescaled
NOVA_ALGOS = ("fedprox", "fedbss", "cqs")             # --nova (FedAvg itself with the FedNova rule: --algo fednova)
PROX_ALGOS = ("fedbss", "cqs")                        # --prox-mu (FedAvg itself with the term: --algo fedprox)
PARTITIONS = ["natural", "motion", "recording"]
DEFAULT_SERVER_LR = {"fedadam": 0.01, "fedavgm": 0.1}
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


# ----------------------------------------------------------------------------- setup
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--partition", default="natural", choices=PARTITIONS)
    ap.add_argument("--algo", default="fedavg", choices=ALGOS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=90, help="centralized epochs (90 = same passes as 30x3)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--client-frac", type=float, default=1.0)
    ap.add_argument("--server-lr", type=float, default=None,
                    help="server step (default 1.0; fedadam 0.01; fedavgm 0.1)")
    ap.add_argument("--server-momentum", type=float, default=0.9, help="fedavgm")
    ap.add_argument("--mu", type=float, default=1e-3, help="fedprox proximal coefficient (on the scaled loss)")
    ap.add_argument("--scaffold-batches", type=int, default=32, help="scaffold: batches for c_i at the server model")
    ap.add_argument("--bss-min", type=float, default=0.6, help="fedbss: always-admitted low-loss fraction")
    ap.add_argument("--bss-warmup", type=int, default=5, help="fedbss: full-data warm-up rounds")
    ap.add_argument("--bss-split", default="quantile", choices=["quantile", "mean"],
                    help="fedbss: low-loss group = fixed fraction --bss-min, or windows with loss <= the client mean")
    ap.add_argument("--cqs-floor", type=float, default=0.6, help="cqs: quota of the first local epoch (q_min)")
    ap.add_argument("--cqs-warmup", type=int, default=5, help="cqs: full-data warm-up rounds (T_w)")
    ap.add_argument("--cqs-pace", default="linear", choices=["linear", "cosine"],
                    help="cqs: quota schedule from the floor to 1 (both end-point anchored; equal for E = 3)")
    ap.add_argument("--cqs-order", default="loss", choices=["loss", "random", "anti"],
                    help="cqs: admission order; random / anti = budget-matched controls")
    ap.add_argument("--cqs-direction", default="ascending", choices=["ascending", "descending"],
                    help="cqs: quotas rising from the floor to 1, or the same quotas in reverse order (same visit "
                         "counts, the highest-loss windows in the first epoch)")
    ap.add_argument("--bn-recal", default="none", choices=["none", "all"],
                    help="re-estimate the BatchNorm statistics of each local model on all its windows after the "
                         "local epochs")
    ap.add_argument("--backbone", default="tcngru", choices=list(BACKBONES),
                    help="tcngru: the CV-anchored predictor of the paper; lstm: LSTM encoder-decoder, no CV anchor")
    ap.add_argument("--static-beta", type=float, default=None, help="--reweight static: alpha_i ~ n_i^beta")
    ap.add_argument("--nova", action="store_true",
                    help="aggregate with the FedNova rule instead of FedAvg (with " + ", ".join(NOVA_ALGOS) + ")")
    ap.add_argument("--prox-mu", type=float, default=None,
                    help="add the FedProx proximal term with this coefficient (with " + ", ".join(PROX_ALGOS) + ")")
    ap.add_argument("--reweight", default="none", choices=["none", "udr", "udr-fs", "static"],
                    help="server reweighting of the sample weights (udr: with " + ", ".join(UDR_ALGOS) + ")")
    ap.add_argument("--udr-temp", type=float, default=1.0, help="udr: temperature tau_d")
    ap.add_argument("--udr-min", type=float, default=0.5, help="udr: lower multiplier bound s_min")
    ap.add_argument("--udr-max", type=float, default=1.5, help="udr: upper multiplier bound s_max")
    ap.add_argument("--tail-frac", type=float, default=0.2, help="fedtr: tail fraction q")
    ap.add_argument("--tail-pos", default="early", choices=["early", "late"], help="fedtr: epoch of the tail")
    ap.add_argument("--tail-weight", default="preserve", choices=["preserve", "one"],
                    help="fedtr: tail loss weight E (objective-preserving) or 1")
    ap.add_argument("--route", default="topk", choices=["topk", "random"], help="fedtr: tail selection")
    ap.add_argument("--tail-warmup", type=int, default=0, help="fedtr: full-data rounds before routing")
    ap.add_argument("--agg", default="lc", choices=["lc", "avg"], help="fedtr: aggregation rule")
    ap.add_argument("--budget", default="adaptive", choices=["adaptive", "fixed", "inverse"],
                    help="fedtr: per-client routing budget q_i (client level)")
    ap.add_argument("--budget-ema", type=float, default=0.7, help="fedtr: EMA factor of the drift signal")
    ap.add_argument("--q-min", type=float, default=0.05, help="fedtr adaptive budget: lower bound of q_i")
    ap.add_argument("--q-max", type=float, default=0.6, help="fedtr adaptive budget: upper bound of q_i")
    ap.add_argument("--precond", default="element", choices=["element", "tensor"],
                    help="fedlc / fedtr: preconditioner per element or one scalar per parameter tensor")
    ap.add_argument("--eval-batch", type=int, default=4096)
    ap.add_argument("--max-windows", type=int, default=0, help="debug: per-client window cap")
    ap.add_argument("--deterministic", type=int, default=0, choices=[0, 1],
                    help="cuDNN deterministic kernels; ~30x slower on an RTX 3050 (dilated Conv1d), so off by "
                         "default. Comparisons are paired by seed within one batch of runs instead.")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per job (several jobs share a server)")
    ap.add_argument("--out-dir", default="runs_fedtraj")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--save-checkpoint", type=int, default=1, choices=[0, 1])
    args = ap.parse_args(argv)
    if args.reweight != "none" and args.algo not in UDR_ALGOS:
        ap.error(f"--reweight {args.reweight} needs FedAvg-type aggregation: --algo {' | '.join(UDR_ALGOS)}")
    if (args.reweight == "static") != (args.static_beta is not None):
        ap.error("--reweight static and --static-beta go together")
    if args.nova and args.algo not in NOVA_ALGOS:
        ap.error(f"--nova needs --algo {' | '.join(NOVA_ALGOS)} (FedAvg with the FedNova rule is --algo fednova)")
    if args.nova and args.reweight != "none":
        ap.error("--nova and --reweight are alternative server rules")
    if args.prox_mu is not None and (args.algo not in PROX_ALGOS or args.prox_mu <= 0):
        ap.error(f"--prox-mu needs a positive value and --algo {' | '.join(PROX_ALGOS)} "
                 "(FedAvg with the proximal term is --algo fedprox)")
    if not 0.0 < args.udr_min <= 1.0 <= args.udr_max:
        ap.error("--udr-min / --udr-max need 0 < s_min <= 1 <= s_max")
    if args.udr_temp <= 0:
        ap.error("--udr-temp must be positive")
    if not 0.0 < args.cqs_floor <= 1.0:
        ap.error("--cqs-floor must lie in (0, 1]")
    if args.cqs_direction != "ascending" and args.algo != "cqs":
        ap.error("--cqs-direction needs --algo cqs")
    if args.bn_recal != "none" and (args.algo == "central" or args.backbone != "tcngru"):
        ap.error("--bn-recal needs a federated algorithm and the tcngru backbone (the LSTM has no BatchNorm)")
    default_slr = DEFAULT_SERVER_LR.get(args.algo, 1.0)
    if args.server_lr is None:
        args.server_lr = default_slr
    if args.run_id is None:
        args.run_id = run_id_of(args, default_slr)
    return args


def run_id_of(args, default_slr):
    """Self-describing run id: every non-default setting that changes the result appears in the name."""
    tag = args.algo
    if args.algo == "fedtr":
        if args.budget != "adaptive":
            tag += f"-b{args.budget}"
        elif (args.q_min, args.q_max, args.budget_ema) != (0.05, 0.6, 0.7):
            tag += f"-qr{args.q_min:g}-{args.q_max:g}-ema{args.budget_ema:g}"
        if args.agg != "lc":
            tag += f"-agg{args.agg}"
        if args.tail_pos != "early":
            tag += f"-{args.tail_pos}"
        if args.tail_weight != "preserve":
            tag += f"-w{args.tail_weight}"
        if args.route != "topk":
            tag += f"-route{args.route}"
        if args.tail_frac != 0.2:
            tag += f"-q{args.tail_frac:g}"
        if args.tail_warmup:
            tag += f"-tw{args.tail_warmup}"
    if args.algo in ("fedlc", "fedtr") and args.precond != "element":
        tag += f"-{args.precond}"
    if args.algo == "fedprox":
        tag += f"-mu{args.mu:g}"
    if args.algo == "fedbss" and (args.bss_min != 0.6 or args.bss_warmup != 5):
        tag += f"-min{args.bss_min:g}-w{args.bss_warmup}"
    if args.algo == "fedbss" and args.bss_split != "quantile":
        tag += f"-{args.bss_split}"
    if args.algo == "cqs":
        if args.cqs_order != "loss":
            tag += f"-{args.cqs_order}"
        if args.cqs_direction != "ascending":
            tag += "-desc"
        if args.cqs_pace != "linear":
            tag += f"-{args.cqs_pace}"
        if args.cqs_floor != 0.6:
            tag += f"-f{args.cqs_floor:g}"
        if args.cqs_warmup != 5:
            tag += f"-w{args.cqs_warmup}"
    if args.reweight == "static":
        tag += f"-static{args.static_beta:g}"
    if args.reweight in ("udr", "udr-fs"):
        tag += "-udr" if args.reweight == "udr" else "-udrfs"
        if args.udr_temp != 1.0:
            tag += f"-t{args.udr_temp:g}"
        if (args.udr_min, args.udr_max) != (0.5, 1.5):
            tag += f"-b{args.udr_min:g}-{args.udr_max:g}"
    if args.nova:
        tag += "-nova"
    if args.prox_mu is not None:
        tag += f"-prox{args.prox_mu:g}"
    if args.server_lr != default_slr:
        tag += f"-slr{args.server_lr:g}"
    if args.client_frac < 1:
        tag += f"-frac{args.client_frac:g}"
    if args.algo == "central":
        if args.epochs != 90:
            tag += f"-e{args.epochs}"
    else:
        if args.rounds != 30:
            tag += f"-r{args.rounds}"
        if args.local_epochs != 3:
            tag += f"-E{args.local_epochs}"
    if args.bn_recal != "none":
        tag += "-bn"
    if args.backbone != "tcngru":
        tag += f"-{args.backbone}"
    if args.lr != 5e-4:
        tag += f"-lr{args.lr:g}"
    if args.max_windows:
        tag += f"-mw{args.max_windows}"
    return f"{tag}_s{args.seed}"


def setup_torch(seed, deterministic, threads):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, threads))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # math attention kernel: the fused kernels have non-deterministic backward passes
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        torch.backends.cudnn.benchmark = True


def gen(device, *keys):
    """Generator seeded by a tuple of integers (seed, round, client, epoch, ...)."""
    h = 0
    for k in keys:
        h = (h * 1_000_003 + int(k) + 7) % (2 ** 62)
    return torch.Generator(device=device).manual_seed(h)


def scaled_loss(model, tf, idx, weight=None):
    """Batch mean of the per-window physical loss (optionally weighted) divided by the CV loss scale.

    Returns (scaled loss, unweighted per-window losses in m^2).
    """
    x, fut, cv = tf.batch(idx)
    e = window_sq_error(tf.predict(model, x, cv), fut)
    if weight is None:
        return e.mean() / tf.loss_scale, e
    return (e * weight).mean() / tf.loss_scale, e


def is_lc(args):
    return args.algo == "fedlc" or (args.algo == "fedtr" and args.agg == "lc")


# ----------------------------------------------------------------------------- local training
def accumulate_precond(opt, params, acc, step):
    """acc += 1 / (sqrt(vhat) + eps) of the AdamW step just taken (its diagonal preconditioner).

    ``step`` is the 1-based number of that step (tracked by the caller to avoid a GPU sync per step).
    """
    bc = 1.0 - ADAM_BETAS[1] ** step
    vs = [opt.state[p]["exp_avg_sq"] for p in params]
    try:                                       # fused multi-tensor kernels: a few launches instead of ~300
        t = torch._foreach_div(vs, bc)
        torch._foreach_sqrt_(t)
        torch._foreach_add_(t, ADAM_EPS)
        torch._foreach_reciprocal_(t)
        torch._foreach_add_(acc, t)
    except (AttributeError, RuntimeError, TypeError):
        for a, v in zip(acc, vs):
            a.add_(torch.reciprocal(torch.sqrt(v / bc) + ADAM_EPS))


def train_epochs(model, tf, epoch_starts, args, seed_keys, epoch_weights=None, prox_ref=None,
                 scaffold_corr=None, opt=None, precond_acc=None):
    """One pass over every tensor in ``epoch_starts`` (shuffled).  A fresh AdamW unless ``opt`` is given.

    ``epoch_weights``: per epoch None or a per-window weight tensor aligned with that epoch's starts.
    Returns (optimizer steps, mean unweighted scaled loss over the visited windows, optimizer).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if opt is None:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=args.weight_decay)
    model.train()
    dev = params[0].device
    steps, n_sum = 0, 0
    loss_acc = torch.zeros((), device=dev)
    mu = args.mu if args.prox_mu is None else args.prox_mu      # proximal coefficient (fedprox, or --prox-mu)
    for e, st in enumerate(epoch_starts):
        w_all = None if epoch_weights is None else epoch_weights[e]
        perm = torch.randperm(st.numel(), generator=gen(dev, *seed_keys, e), device=dev)
        for b in range(0, st.numel(), args.batch_size):
            sel = perm[b:b + args.batch_size]
            loss, err = scaled_loss(model, tf, st[sel], None if w_all is None else w_all[sel])
            total = loss
            if prox_ref is not None:
                total = total + 0.5 * mu * sum(((p - q) ** 2).sum() for p, q in zip(params, prox_ref))
            opt.zero_grad(set_to_none=True)
            total.backward()
            if scaffold_corr is not None:
                for p, c in zip(params, scaffold_corr):
                    p.grad.add_(c)
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            if precond_acc is not None:
                accumulate_precond(opt, params, precond_acc, steps + 1)
            steps += 1
            loss_acc += err.detach().sum() / tf.loss_scale
            n_sum += sel.numel()
    return steps, float(loss_acc) / max(n_sum, 1), opt


def gradient_at(model, tf, starts, args, max_batches, seed_keys):
    """Mean gradient of the scaled loss at the current parameters on up to ``max_batches`` random batches."""
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()
    dev = params[0].device
    grads = [torch.zeros_like(p) for p in params]
    perm = torch.randperm(starts.numel(), generator=gen(dev, *seed_keys, 77), device=dev)
    nb = 0
    for b in range(0, min(starts.numel(), max_batches * args.batch_size), args.batch_size):
        g = torch.autograd.grad(scaled_loss(model, tf, starts[perm[b:b + args.batch_size]])[0], params)
        for a, gi in zip(grads, g):
            a.add_(gi)
        nb += 1
    return [a / max(nb, 1) for a in grads]


@torch.no_grad()
def recalibrate_bn(model, tf, starts, args, seed_keys):
    """Re-estimate the BatchNorm running statistics of ``model`` on the windows ``starts`` (``--bn-recal all``).

    One pass in training mode (dropout active, as when the statistics are collected during training) in shuffled
    mini-batches of the training batch size, without gradients, with the cumulative average over the pass
    (momentum None) instead of the exponential average.  The momentum is restored afterwards.  Returns the number
    of BatchNorm layers.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    moms = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    perm = torch.randperm(starts.numel(), generator=gen(starts.device, *seed_keys, 55), device=starts.device)
    for b in range(0, starts.numel(), args.batch_size):
        x, _, _ = tf.batch(starts[perm[b:b + args.batch_size]])
        model(x)
    for m, mo in zip(bns, moms):
        m.momentum = mo
    return len(bns)


def cqs_quotas(E, floor, pace="linear"):
    """Admitted fractions q_1..q_E of the loss-ordered windows under CQS (end points floor and 1).

    linear: q_e = floor + (1 - floor) (e - 1) / (E - 1).  cosine: the end-point anchored cosine
    floor + (1 - floor) [1 - cos(pi (e - 1) / (E - 1))] / 2.  Both give floor, (1 + floor) / 2, 1 for E = 3 and
    differ only for E >= 4.  E = 1 admits every window.
    """
    if E <= 1:
        return [1.0]
    out = []
    for e in range(E):
        prog = e / (E - 1)
        if pace == "cosine":
            prog = 0.5 * (1.0 - math.cos(math.pi * prog))
        out.append(floor + (1.0 - floor) * prog)
    return out


def admitted_count(n, q):
    """ceil(n q) windows, at least one; the tolerance keeps a product such as 8.000000000000002 at 8."""
    return max(1, min(n, int(math.ceil(n * q - 1e-9))))


def client_schedule(args, rnd, starts, global_model, tf, client_id, q=None):
    """Per-epoch window sets, per-epoch window weights (None = 1) and the number of routed / deferred windows.

    ``q``: routing budget of this client (fedtr); defaults to ``--tail-frac``.
    """
    E = args.local_epochs
    n = starts.numel()
    if args.algo == "cqs" and rnd >= args.cqs_warmup:
        if args.cqs_order == "random":
            # budget-matched control: a fresh random order per client and round; it needs no scoring pass
            order = torch.randperm(n, generator=gen(starts.device, args.seed, rnd, client_id, 98),
                                   device=starts.device)
        else:
            scores = score_windows(global_model, tf, starts, args.eval_batch)
            # stable sort: ties keep the window order (torch.sort has `stable` since 1.9, argsort only later)
            order = torch.sort(scores, descending=args.cqs_order == "anti", stable=True).indices
        # one ranking per round, reused by every local epoch; epoch e trains on the first ceil(n q_e) windows
        quotas = cqs_quotas(E, args.cqs_floor, args.cqs_pace)
        if args.cqs_direction == "descending":
            quotas = quotas[::-1]          # same prefixes of the same ranking, so the same visit counts
        out = [starts[order[:admitted_count(n, qe)]] for qe in quotas]
        # windows left out of at least one epoch (the first epoch under rising quotas, as before)
        return out, [None] * E, int(n - min(o.numel() for o in out))
    if args.algo == "fedbss" and rnd >= args.bss_warmup:
        scores = score_windows(global_model, tf, starts, args.eval_batch)
        order = torch.argsort(scores)
        q0 = args.bss_min          # low-loss group that every epoch admits
        if args.bss_split == "mean":
            q0 = float((scores <= scores.mean()).float().mean())
        out = []
        for e in range(1, E + 1):
            q = q0 + (1 - q0) * (1 - math.cos(math.pi * e / E)) / 2
            out.append(starts[order[:max(1, int(math.ceil(n * q)))]])
        return out, [None] * E, int(n - out[0].numel())
    if args.algo == "fedtr" and rnd >= args.tail_warmup and E > 1:
        k = int(round((args.tail_frac if q is None else q) * n))
        tail = torch.zeros(n, dtype=torch.bool, device=starts.device)
        if args.route == "random":
            perm = torch.randperm(n, generator=gen(starts.device, args.seed, rnd, client_id, 99), device=starts.device)
            tail[perm[:k]] = True
        else:
            tail[torch.topk(score_windows(global_model, tf, starts, args.eval_batch), k).indices] = True
        body = starts[~tail]
        w = None
        if args.tail_weight == "preserve":
            w = torch.ones(n, device=starts.device)
            w[tail] = float(E)
        if args.tail_pos == "early":
            return [starts] + [body] * (E - 1), [w] + [None] * (E - 1), k
        return [body] * (E - 1) + [starts], [None] * (E - 1) + [w], k
    return [starts] * E, [None] * E, 0


# ----------------------------------------------------------------------------- aggregation
def update_deviations(D):
    """UDR deviations d_i = ||Delta_i - (1/m) sum_j Delta_j||_2 of flattened trainable-parameter updates.

    The reference is the unweighted cohort mean on purpose: a sample-weighted mean would be dominated by the
    largest clients and understate their own deviation.
    """
    mean = torch.stack(D).mean(dim=0)
    return np.array([float((d - mean).norm()) for d in D], dtype=np.float64)


def udr_multipliers(dev, p, temp=1.0, s_min=0.5, s_max=1.5, eps=1e-12):
    """Update-deviation reweighting: aggregation weights alpha from deviations ``dev`` and sample shares ``p``.

    r_i = exp(-d_i / (m tau_d)) with m = max(median d, eps); s_i = clip(r_i / sum_j p_j r_j, s_min, s_max);
    alpha_i = p_i s_i / sum_j p_j s_j.  Normalizing by the sample-weighted mean makes the multipliers average to
    one under the FedAvg weights before clipping, so alpha_i / p_i lies in [s_min / s_max, s_max / s_min], equal
    deviations give alpha = p, and dividing by the median makes the rule invariant to a common rescaling of
    the updates.  Returns (alpha, s).
    """
    dev = np.asarray(dev, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if dev.size <= 1:
        return p.copy(), np.ones_like(p)
    m = max(float(np.median(dev)), eps)
    r = np.exp(-dev / (m * temp))
    s = np.clip(r / max(float(np.dot(p, r)), eps), s_min, s_max)
    alpha = p * s / float(np.dot(p, s))
    return alpha, s


class Server:
    def __init__(self, model, args):
        self.args = args
        self.param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.state_keys = list(model.state_dict().keys())
        self.mom = None          # fedavgm / fedadam first moment
        self.var = None          # fedadam second moment
        self.c = None            # scaffold global control variate
        self.ema_cos = {}        # fedtr adaptive budget: client id -> EMA of leave-one-out update alignment

    def aggregate(self, gstate, clients):
        """``clients``: list of dicts with n, state, steps (and N, the accumulated preconditioner, for lc)."""
        a = self.args
        n = np.array([c["n"] for c in clients], dtype=np.float64)
        p = n / n.sum()
        tau = np.array([c["steps"] for c in clients], dtype=np.float64)
        tau_eff = float((p * tau).sum())
        new = {}
        deltas = {k: [c["state"][k] - gstate[k] for c in clients] for k in self.param_names}
        D = [torch.cat([deltas[k][i].flatten() for k in self.param_names]) for i in range(len(clients))]
        diag = {"tau_eff": tau_eff, "p": p.tolist(), "steps": tau.tolist()}
        # client weights of the FedAvg-type rules: the sample shares, or the UDR weights (parameters and buffers)
        wt = p
        if a.reweight == "static":
            wt = n ** a.static_beta / float(np.sum(n ** a.static_beta))
            diag["alpha"] = [round(float(v), 6) for v in wt]
        if a.reweight in ("udr", "udr-fs"):
            DD = D
            if a.reweight == "udr-fs":         # full state: the floating buffers join the deviation
                fkeys = [k for k in self.state_keys if k not in self.param_names and torch.is_floating_point(gstate[k])]
                DD = [torch.cat([D[i]] + [(c["state"][k] - gstate[k]).flatten() for k in fkeys])
                      for i, c in enumerate(clients)]
            dev = update_deviations(DD)
            wt, s = udr_multipliers(dev, p, a.udr_temp, a.udr_min, a.udr_max)
            diag["udr_dev"] = [float(f"{v:.6g}") for v in dev]
            diag["udr_s"] = [round(float(v), 5) for v in s]
            diag["alpha"] = [round(float(v), 6) for v in wt]
        # buffers: weighted average with wt, which is the sample share except under UDR (integer buffers copied
        # from the first client)
        for k in self.state_keys:
            if k in self.param_names:
                continue
            t0 = clients[0]["state"][k]
            if torch.is_floating_point(t0):
                new[k] = sum(float(wi) * c["state"][k] for wi, c in zip(wt, clients))
            else:
                new[k] = t0.clone()
        if is_lc(a):
            logw = np.zeros(len(clients))
            size = 0
            for j, k in enumerate(self.param_names):
                inv = [float(pi) / c["N"][j] for pi, c in zip(p, clients)]       # p_i / N_i
                tot = sum(inv)                                                   # = 1 / N_s
                new[k] = gstate[k] + a.server_lr * sum((w / tot) * d for w, d in zip(inv, deltas[k]))
                numel = gstate[k].numel()
                for i, w in enumerate(inv):          # weight relative to the sample share, log-averaged
                    r = torch.log((w / tot) / float(p[i]))
                    logw[i] += float(r.sum()) if r.numel() > 1 else float(r) * numel
                size += numel
            diag["rel_weight"] = np.exp(logw / size).tolist()
        elif a.algo == "fednova" or a.nova:          # --nova: the same rule on top of fedprox / fedbss / cqs
            for k in self.param_names:
                acc = sum(float(pi) * d / ti for pi, d, ti in zip(p, deltas[k], tau))
                new[k] = gstate[k] + a.server_lr * tau_eff * acc
        elif a.algo == "fedavgm":
            avg = {k: sum(float(pi) * d for pi, d in zip(p, deltas[k])) for k in self.param_names}
            if self.mom is None:
                self.mom = {k: torch.zeros_like(v) for k, v in avg.items()}
            for k in self.param_names:
                self.mom[k] = a.server_momentum * self.mom[k] + avg[k]
                new[k] = gstate[k] + a.server_lr * self.mom[k]
        elif a.algo == "fedadam":
            b1, b2, eps = 0.9, 0.99, 1e-3
            if self.mom is None:
                self.mom = {k: torch.zeros_like(gstate[k]) for k in self.param_names}
                self.var = {k: torch.zeros_like(gstate[k]) for k in self.param_names}
            for k in self.param_names:
                g = -sum(float(pi) * d for pi, d in zip(p, deltas[k]))
                self.mom[k] = b1 * self.mom[k] + (1 - b1) * g
                self.var[k] = b2 * self.var[k] + (1 - b2) * g * g
                new[k] = gstate[k] - a.server_lr * self.mom[k] / (self.var[k].sqrt() + eps)
        else:   # fedavg, fedprox, fedbss, cqs, scaffold, fedtr with --agg avg (wt = p unless --reweight udr)
            for k in self.param_names:
                new[k] = gstate[k] + a.server_lr * sum(float(wi) * d for wi, d in zip(wt, deltas[k]))
            diag["keff"] = float(1.0 / np.sum(np.square(wt)))     # effective number of clients
        # diagnostics: local update norms, alignment with the aggregate, and alignment with the sample-weighted
        # mean update of the *other* clients (leave-one-out; drives the adaptive routing budget, so that a
        # dominant client is not judged against an average it defines itself)
        agg = torch.cat([(new[k] - gstate[k]).flatten() for k in self.param_names])
        S = sum(float(pi) * d for pi, d in zip(p, D))
        norms, cosines, loo = [], [], []
        for i, d in enumerate(D):
            norms.append(float(d.norm()))
            cosines.append(float(torch.dot(d, agg) / (d.norm() * agg.norm() + 1e-12)))
            if len(D) == 1 or p[i] > 1 - 1e-9:
                loo.append(1.0)
            else:
                m = (S - float(p[i]) * d) / (1.0 - float(p[i]))
                loo.append(float(torch.dot(d, m) / (d.norm() * m.norm() + 1e-12)))
        diag["delta_norm"] = norms
        diag["cos_to_agg"] = cosines
        diag["cos_loo"] = loo
        diag["agg_norm"] = float(agg.norm())
        return new, diag

    # ------------------------------------------------------------------ client-level routing budget (fedtr)
    def budgets(self, p_all):
        """Routing budget q_i of every client for the next round.

        With routing budget q_i, a window of client i is trained s_i = 1 + (E-1)(1-q_i) times per round on
        average.  The second-order (drift) term of local SGD grows like s_i^2 zeta_i^2, zeta_i^2 being the
        client's gradient dissimilarity.  Minimizing sum_i p_i s_i^2 zeta_i^2 at a fixed compute budget
        sum_i p_i s_i = s_bar (s_bar = 1 + (E-1)(1-q), the fixed-budget value) gives s_i proportional to
        1 / zeta_i^2.  zeta_i^2 is estimated by d_i = 1 - EMA(cos_loo_i) (for updates of similar norm,
        ||u_i - u_bar||^2 = 2 ||u||^2 (1 - cos)).  The bounds [q_min, q_max] are enforced by solving for the
        scale of s_i by bisection, so the budget constraint holds after clipping.

        fixed: q_i = q.  adaptive: s_i proportional to 1 / d_i.  inverse: s_i proportional to d_i (direction
        check).  Clients never observed use the mean drift.
        """
        a = self.args
        K = len(p_all)
        E = a.local_epochs
        if a.algo != "fedtr" or a.budget == "fixed" or not self.ema_cos or E < 2:
            return [a.tail_frac] * K
        known = {i: max(1.0 - c, 1e-3) for i, c in self.ema_cos.items()}
        fill = float(np.mean(list(known.values())))
        d = np.array([known.get(i, fill) for i in range(K)], dtype=np.float64)
        w = 1.0 / d if a.budget == "adaptive" else d
        s_bar = 1.0 + (E - 1) * (1.0 - a.tail_frac)
        s_lo = 1.0 + (E - 1) * (1.0 - a.q_max)
        s_hi = 1.0 + (E - 1) * (1.0 - a.q_min)
        lo, hi = 0.0, 2.0 * s_hi / w.min() + 1.0
        for _ in range(80):
            c = 0.5 * (lo + hi)
            if float(np.dot(p_all, np.clip(c * w, s_lo, s_hi))) < s_bar:
                lo = c
            else:
                hi = c
        s = np.clip(0.5 * (lo + hi) * w, s_lo, s_hi)
        return (1.0 - (s - 1.0) / (E - 1)).tolist()

    def observe(self, cohort, cos_loo):
        b = self.args.budget_ema
        for c, v in zip(cohort, cos_loo):
            self.ema_cos[c] = v if c not in self.ema_cos else b * self.ema_cos[c] + (1.0 - b) * v


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_torch(args.seed, args.deterministic, args.threads)
    out = Path(args.out_dir) / Path(args.data_dir).name / args.partition / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    store = WindowStore(args.data_dir, args.partition, dev, max_windows=args.max_windows or None)
    tf = FeatureTransform(store, anchor="none" if args.backbone == "lstm" else "cv")
    torch.manual_seed(args.seed)
    gmodel = build_model(backbone=args.backbone).to(dev)
    K = store.num_clients
    n_train = store.counts("train")
    print(f"[{args.run_id}] data={store.dir.name} partition={args.partition} clients={K} "
          f"train windows={n_train} backbone={args.backbone} params={count_parameters(gmodel)} device={dev}",
          flush=True)

    fields = ["round", "time_s", "train_loss", "val_1s", "val_2s", "val_3s", "val_4s", "val_5s", "val_mean",
              "val_loss_m2", "visits", "tail", "extra"]
    fh = open(out / "log.csv", "w", newline="", encoding="utf-8")
    fh.write("# " + json.dumps(vars(args)) + "\n")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    best = {"val_5s": float("inf"), "round": -1}
    curve = []
    stop = {"diverged": 0}

    def checkpoint(rnd, row):
        v = evaluate(gmodel, tf, "valid", args.eval_batch)
        if not math.isfinite(v["5s"]):
            # a diverged configuration is a result (e.g. too large a server step), not a crash
            stop["diverged"] = rnd
            print(f"[{args.run_id}] round {rnd}: non-finite validation error, stopping", flush=True)
        row.update({f"val_{h}": v[h] for h in ("1s", "2s", "3s", "4s", "5s")})
        row["val_mean"], row["val_loss_m2"] = v["mean"], v["loss_m2"]
        curve.append({"round": rnd, **{h: v[h] for h in ("1s", "2s", "3s", "4s", "5s", "mean")}})
        if v["5s"] < best["val_5s"]:
            best.update(val_5s=v["5s"], round=rnd)
            if args.save_checkpoint:
                torch.save({"model": gmodel.state_dict(), "round": rnd, "args": vars(args), "tf": tf.state()},
                           out / "best.pt")
            else:
                best["state"] = {k: t.detach().clone() for k, t in gmodel.state_dict().items()}
        writer.writerow(row)
        fh.flush()
        print(f"[{args.run_id}] round {rnd:3d}  val 5s {v['5s']:.4f}  mean {v['mean']:.4f}  "
              f"train {row['train_loss']:.4f}  {row['time_s']:.1f}s", flush=True)

    if args.algo == "central":
        params = [p for p in gmodel.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=args.weight_decay)
        tr = store.starts["train"]
        for ep in range(args.epochs):
            t0 = time.time()
            steps, loss, opt = train_epochs(gmodel, tf, [tr], args, (args.seed, ep, 0), opt=opt)
            checkpoint(ep + 1, {"round": ep + 1, "time_s": time.time() - t0, "train_loss": loss,
                                "visits": tr.numel(), "tail": 0, "extra": json.dumps({"steps": steps})})
            if stop["diverged"]:
                break
    else:
        server = Server(gmodel, args)
        lmodel = build_model(backbone=args.backbone).to(dev)
        params_l = [p for p in lmodel.parameters() if p.requires_grad]
        client_starts = [store.client_starts("train", c) for c in range(K)]
        c_local = None
        if args.algo == "scaffold":
            c_local = [[torch.zeros_like(p) for p in params_l] for _ in range(K)]
            server.c = [torch.zeros_like(p) for p in params_l]
        m = max(1, int(round(args.client_frac * K)))
        p_all = np.array(n_train, dtype=np.float64) / float(sum(n_train))
        for rnd in range(args.rounds):
            t0 = time.time()
            gstate = {k: v.detach().clone() for k, v in gmodel.state_dict().items()}
            if m < K:
                cohort = sorted(np.random.default_rng(args.seed * 100003 + rnd).choice(K, m, replace=False))
            else:
                cohort = list(range(K))
            results, visits, tails, losses = [], 0, 0, []
            q_all = server.budgets(p_all)
            for c in cohort:
                st = client_starts[c]
                epochs, weights, n_tail = client_schedule(args, rnd, st, gmodel, tf, c, q=q_all[c])
                lmodel.load_state_dict(gstate)
                prox_ref = ([p.detach().clone() for p in params_l]
                            if args.algo == "fedprox" or args.prox_mu is not None else None)
                corr, c_new = None, None
                if args.algo == "scaffold":
                    c_new = gradient_at(lmodel, tf, st, args, args.scaffold_batches, (args.seed, rnd, c))
                    lmodel.load_state_dict(gstate)                 # undo BatchNorm statistics updates
                    corr = [cg - ci for cg, ci in zip(server.c, c_local[c])]
                acc = [torch.zeros_like(p) for p in params_l] if is_lc(args) else None
                steps, loss, opt = train_epochs(lmodel, tf, epochs, args, (args.seed, rnd, c), epoch_weights=weights,
                                                prox_ref=prox_ref, scaffold_corr=corr, precond_acc=acc)
                del opt
                if args.bn_recal == "all":
                    recalibrate_bn(lmodel, tf, st, args, (args.seed, rnd, c))
                res = {"n": st.numel(), "steps": steps,
                       "state": {k: v.detach().clone() for k, v in lmodel.state_dict().items()}}
                if acc is not None:
                    res["N"] = [args.lr * (t.mean() if args.precond == "tensor" else t) for t in acc]
                if args.algo == "scaffold":
                    res["dc"] = [a - b for a, b in zip(c_new, c_local[c])]
                    c_local[c] = c_new
                results.append(res)
                visits += sum(int(e.numel()) for e in epochs)
                tails += n_tail
                losses.append(loss)
            new_state, diag = server.aggregate(gstate, results)
            if args.algo == "scaffold":
                nn_ = np.array([r["n"] for r in results], dtype=np.float64)
                pp = nn_ / nn_.sum()
                for j in range(len(server.c)):
                    server.c[j] = server.c[j] + (m / K) * sum(float(pi) * r["dc"][j] for pi, r in zip(pp, results))
            gmodel.load_state_dict(new_state)
            del results
            server.observe(cohort, diag["cos_loo"])
            diag["cohort"] = [int(c) for c in cohort]
            diag["client_loss"] = losses
            diag["q"] = [float(q_all[c]) for c in cohort]
            checkpoint(rnd + 1, {"round": rnd + 1, "time_s": time.time() - t0,
                                 "train_loss": float(np.average(losses, weights=[client_starts[c].numel()
                                                                                for c in cohort])),
                                 "visits": visits, "tail": tails, "extra": json.dumps(diag)})
            if stop["diverged"]:
                break
    fh.close()

    # ------------------------------------------------------------------ test at the best validation round
    last = evaluate(gmodel, tf, "test", args.eval_batch)
    if best["round"] > 0:
        if args.save_checkpoint:
            gmodel.load_state_dict(torch.load(out / "best.pt", map_location=dev, weights_only=False)["model"])
        else:
            gmodel.load_state_dict(best["state"])
    test = evaluate(gmodel, tf, "test", args.eval_batch, keep_windows=True)
    strata = stratify(test)
    for k in ("_w5", "_cv5", "_lat5"):
        test.pop(k)
    result = {"run_id": args.run_id, "args": vars(args), "best_round": best["round"],
              "best_val_5s": best["val_5s"], "test": test, "test_last_round": last, "strata": strata,
              "client_names": store.client_names, "train_counts": n_train, "val_curve": curve,
              "cv_reference": store.meta["cv_rmse"], "params": count_parameters(gmodel),
              "diverged_round": stop["diverged"],
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "wall_s": time.time() - t_start}
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[{args.run_id}] best round {best['round']}  test 5s {test['5s']:.4f}  mean {test['mean']:.4f}  "
          f"(CV 5s {store.meta['cv_rmse']['test']['5s']:.4f})  wall {result['wall_s'] / 60:.1f} min", flush=True)
    return result


if __name__ == "__main__":
    main()
