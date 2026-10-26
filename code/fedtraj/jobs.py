"""Experiment stages: which training runs make up each stage.  Finished runs (result.json) are skipped, so a
stage can be restarted at any time, and runs shared by two stages are trained once.

Method names in the paper -> configurations here
    FedTR (proposed)         --algo fedtr            tail routing (sample) + drift-adaptive budget (client)
                                                     + consistent aggregation of Wang et al. 2021 (component)
    w/o client-level budget  --algo fedtr --budget fixed
    sample level only        --algo fedtr --budget fixed --agg avg
    borrowed rule only       --algo fedlc            (local + global correction, no routing)
    ablations                --budget inverse | --agg avg | --route random | --tail-pos late | --tail-weight one
                             | --precond tensor | --tail-frac 0.1 / 0.3

Fed-SSBR (the current paper) -> configurations here
    CQS                      --algo cqs              curriculum-quota scheduling (client layer)
    UDR alone                --algo fedavg --reweight udr
    Fed-SSBR (CQS + UDR)     --algo cqs --reweight udr
    CQS-random               --algo cqs --cqs-order random   budget-matched control of the admission order
    FedBSS-pace              --algo fedbss

Stages (run them in this order; ``launch.py`` executes a stage)
    smoke     every configuration for one tiny round on both datasets (``runs_smoke/``), ~5 min
    timing    3 full rounds of fedavg / fednova / fedbss / fedlc / fedtr on NGSIM-natural, one job per GPU
              (``runs_timing/``): round times for the cost comparison
    pilot     seed 1 on NGSIM-natural, highD-natural, highD-motion: central, fedavg, fednova, fedlc, fedtr,
              fedtr with a fixed budget, sample level only, fedtr with 5 warm-up rounds.  Decision gate before
              the main study (random routing is tested in the ablation stage).
    tune      seed 1 on both natural partitions: FedProx mu, FedAdam / FedAvgM / FedAvg server lr.
              ``python -m fedtraj.jobs --select-tuned`` then writes runs_fedtraj/tuned.json (by validation).
    main      seeds 1-5 on both natural partitions: all baselines (tuned), fedlc, fedtr
    ablation  seeds 1-5 on both natural partitions: fixed budget, sample level only, inverse budget, random
              routing; seeds 1-3 on NGSIM-natural: adaptive budget with FedAvg aggregation, late tail, tail
              weight 1, per-tensor preconditioner, tail fraction 0.1 / 0.3
    robust    seeds 1-3: motion partitions; half participation on both natural partitions; the 60 highD
              recordings as clients with 10% participation and 100 rounds (each with fixed and adaptive budget)

Fed-SSBR stages (after smoke and timing, which include the CQS / UDR configurations)
    ssbr-pilot   seed 1 on the two main settings, NGSIM by recording period (natural, 6 clients) and highD by
                 motion pattern (motion, 10 clients): central, FedAvg, FedProx (mu 1e-4 ... 1), FedNova, FedLC,
                 FedBSS-pace, CQS, UDR, Fed-SSBR, CQS-random.  Decision gate before the main study:
                 ``--ssbr-select`` prints the check and writes runs_fedtraj/tuned_ssbr.json (FedProx mu)
    ssbr-main    seeds 1-5 on the two main settings, the same ten configurations with the selected mu; then
                 seeds 6-8 for the five key configurations (FedAvg, CQS, Fed-SSBR, CQS-random, FedProx with the
                 selected mu), fixed on 2026-10-22 before any main result was seen
    ssbr-robust  seeds 1-3: NGSIM by motion pattern (the partition type of the original study): FedAvg, CQS,
                 CQS-random, UDR, Fed-SSBR; the 60 highD recordings as clients with 10% participation and
                 100 rounds: FedAvg, FedNova, CQS, UDR, Fed-SSBR
    ssbr-extra   the remaining controls of the paper on the two main settings.  Seeds 1-5: CQS with static weights
                 n_i^beta (beta matched to the mean K_eff of Fed-SSBR in the pilot), FedBSS-mean, CQS + UDR-FS.
                 Seeds 1-3: CQS-anti; half participation for FedAvg, CQS, UDR and Fed-SSBR.  Seed 1: sensitivity
                 of Fed-SSBR (quota floor, warm-up, UDR temperature, multiplier bounds) and two same-seed replicates
                 of the default.  NGSIM, seeds 1-3: linear and cosine quotas at E = 5.  Needs tuned_ssbr.json with
                 static_beta (written by ``--ssbr-select``)
    ssbr-combo   added on 2026-10-24, after the main results: CQS combined with existing rules instead of UDR,
                 CQS + FedNova (step-normalized aggregation) and CQS + FedProx (the mu selected in the pilot, not
                 re-tuned), seeds 1-5 on the two main settings.  The seeds were fixed before any of these runs and
                 every result is reported

Checks for the revision after R16 (fixed on 2026-10-26, after all results of R16 were known, before any of
these runs; every run is reported).  Output in ``runs_r17/`` on a second server (two RTX 4090), the two main
settings, all other settings as in the paper:
    r17-order       admission timing at fixed visit counts.  Seeds 1-8: CQS and CQS with descending quotas
                    (1.0, 0.8, 0.6), both with the BatchNorm statistics re-estimated on all windows after the local
                    epochs (``--bn-recal all``), so that the two differ only in when the windows are used.  Seeds
                    1-3: descending quotas without re-estimation (size of the BatchNorm effect) and CQS exactly as
                    in the paper (re-estimation effect and agreement with the first server).
    r17-lstm-pilot  second backbone (``--backbone lstm``, no CV anchor): FedAvg, seed 1, learning rate 5e-4,
                    1e-3, 2e-3.  ``--r17-select`` picks per setting the rate with the lowest best validation 5 s
                    RMSE of FedAvg and writes runs_r17/tuned_r17.json (the rate is chosen on the baseline only)
    r17-lstm        seeds 1-5 with the selected rate: FedAvg, CQS, FedBSS-fixed, CQS-random
    r17-main        r17-order followed by r17-lstm (r17-order alone while tuned_r17.json is missing)

Usage
    python -m fedtraj.jobs --stage pilot              # print the remaining commands of a stage
    python -m fedtraj.jobs --select-tuned             # after the tune stage
    python -m fedtraj.jobs --ssbr-select              # after the ssbr-pilot stage
    python -m fedtraj.jobs --r17-select               # after the r17-lstm-pilot stage
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STAGES = ["smoke", "timing", "pilot", "tune", "main", "ablation", "robust", "ssbr-pilot", "ssbr-main",
          "ssbr-robust", "ssbr-extra", "ssbr-combo", "r17-smoke", "r17-order", "r17-lstm-pilot", "r17-lstm",
          "r17-main"]
TUNE_GRID = {
    "fedprox": ("--mu", ["1e-4", "1e-3", "1e-2"]),
    "fedadam": ("--server-lr", ["0.003", "0.01", "0.03"]),
    "fedavgm": ("--server-lr", ["0.1", "0.3", "1.0"]),
    "fedavg": ("--server-lr", ["1.0", "1.5", "2.0"]),
}
STAGE_OUT = {"smoke": "runs_smoke", "timing": "runs_timing", "r17-smoke": "runs_r17_smoke",
             "r17-order": "runs_r17", "r17-lstm-pilot": "runs_r17", "r17-lstm": "runs_r17", "r17-main": "runs_r17"}
OURS = ["--algo", "fedtr"]                                   # both levels: routing + adaptive budget + lc
FIXED = OURS + ["--budget", "fixed"]                          # without the client-level budget
SAMPLE_ONLY = OURS + ["--budget", "fixed", "--agg", "avg"]    # sample level only, FedAvg aggregation
CLIENT_ONLY = ["--algo", "fedlc"]                             # borrowed aggregation rule only (no routing)

# Fed-SSBR
CQS = ["--algo", "cqs"]                                       # client layer
UDR = ["--algo", "fedavg", "--reweight", "udr"]               # server layer alone
SSBR = ["--algo", "cqs", "--reweight", "udr"]                 # both layers
CQS_RANDOM = ["--algo", "cqs", "--cqs-order", "random"]       # same visit counts, random admission order
# FedProx grid of the pilot.  1e-1 and 1 were added after the first pilot (2026-10-22), in which 1e-2, then the
# largest value, gave the lowest validation error in both main settings: a baseline tuned at the grid boundary.
SSBR_MU = ["1e-4", "1e-3", "1e-2", "1e-1", "1"]
SSBR_GATE_M = 0.02                                            # pilot gate: CQS below FedAvg by this much
# Seeds 6-8 for the key configurations: the pilot gains at the best-validation checkpoint were about 0.02-0.045 m,
# and with five seeds the 95% interval half-width is 1.24 s_d (eight seeds: 0.84 s_d).  Fixed before ssbr-main.
SSBR_KEY_SEEDS = range(6, 9)

# Checks after R16
BN = ["--bn-recal", "all"]                                    # BatchNorm statistics re-estimated on all windows
CQS_DESC = CQS + ["--cqs-direction", "descending"]            # quotas 1.0, 0.8, 0.6: same counts, reversed timing
# second backbone without the CV anchor; evaluation and scoring in batches of 1024 windows, because the cuDNN LSTM
# needs about 2.4 GiB of workspace for a batch of 4096 (the batch size does not change the per-window predictions)
LSTM = ["--backbone", "lstm", "--eval-batch", "1024"]
R17_LSTM_LR = ["5e-4", "1e-3", "2e-3"]                        # pilot grid of the LSTM learning rate (FedAvg)


def ssbr_settings(NG, HD):
    """Main settings of the Fed-SSBR study: NGSIM by recording period (6 clients), highD by motion (10)."""
    return ((NG, "natural"), (HD, "motion"))


def ssbr_configs(mus):
    """The ten configurations of the Fed-SSBR main study, with one FedProx run per value in ``mus``."""
    return ([["--algo", "central"], ["--algo", "fedavg"]] + [["--algo", "fedprox", "--mu", m] for m in mus]
            + [["--algo", "fednova"], ["--algo", "fedlc"], ["--algo", "fedbss"], CQS, UDR, SSBR, CQS_RANDOM])


def ssbr_key_configs(mu):
    """The five key configurations that get the extra seeds: FedAvg, CQS, Fed-SSBR, CQS-random, FedProx (mu)."""
    return [["--algo", "fedavg"], CQS, SSBR, CQS_RANDOM, ["--algo", "fedprox", "--mu", mu]]


def setting_key(data, part):
    return f"{Path(data).name}/{part}"


def specs(stage, NG, HD, tuned=None):
    """Yield (data_dir, partition, seed, extra argv) for a stage."""
    if stage == "smoke":
        tiny = ["--rounds", "1", "--local-epochs", "2", "--max-windows", "1500", "--save-checkpoint", "0"]
        configs = [["--algo", a] for a in ("fedavg", "fedprox", "fednova", "fedavgm", "fedadam", "scaffold")]
        configs += [["--algo", "fedbss", "--bss-warmup", "0"], CLIENT_ONLY, OURS, FIXED, SAMPLE_ONLY,
                    OURS + ["--budget", "inverse", "--route", "random", "--tail-pos", "late", "--tail-weight", "one",
                            "--precond", "tensor"]]
        # Fed-SSBR, with warm-up 0 so that the single smoke round runs the scheduler
        configs += [CQS + ["--cqs-warmup", "0"], CQS_RANDOM + ["--cqs-warmup", "0"],
                    CQS + ["--cqs-warmup", "0", "--cqs-order", "anti"], UDR, SSBR + ["--cqs-warmup", "0"]]
        for data in (NG, HD):
            for e in configs:
                yield data, "natural", 1, e + tiny
            yield data, "natural", 1, ["--algo", "central", "--epochs", "1", "--max-windows", "1500",
                                       "--save-checkpoint", "0"]
        yield HD, "recording", 1, OURS + ["--client-frac", "0.1"] + tiny
        yield HD, "recording", 1, SSBR + ["--cqs-warmup", "0", "--client-frac", "0.1"] + tiny
    elif stage == "timing":
        t = ["--rounds", "3", "--save-checkpoint", "0"]
        # CQS-random skips the scoring pass, so CQS minus CQS-random is the scoring cost
        for e in (["--algo", "fedavg"], ["--algo", "fednova"], ["--algo", "fedbss", "--bss-warmup", "0"],
                  CLIENT_ONLY, OURS, CQS + ["--cqs-warmup", "0"], CQS_RANDOM + ["--cqs-warmup", "0"],
                  SSBR + ["--cqs-warmup", "0"]):
            yield NG, "natural", 1, e + t
    elif stage == "pilot":
        # fedtr-tw5: routing starts after 5 full-data rounds (as FedBSS); a local reduced-data run showed that
        # routing from the CV initialization slows the first rounds.  The better variant (by validation) becomes
        # the default of the later stages.
        configs = [["--algo", "central"], ["--algo", "fedavg"], ["--algo", "fednova"], CLIENT_ONLY, OURS,
                   FIXED, SAMPLE_ONLY, OURS + ["--tail-warmup", "5"]]
        for data, part in ((NG, "natural"), (HD, "natural"), (HD, "motion")):
            for e in configs:
                yield data, part, 1, e
    elif stage == "tune":
        for data in (NG, HD):
            for algo, (flag, values) in TUNE_GRID.items():
                for v in values:
                    yield data, "natural", 1, ["--algo", algo, flag, v]
    elif stage == "main":
        if tuned is None:
            raise SystemExit("main stage needs runs_fedtraj/tuned.json: run the tune stage, then "
                             "`python -m fedtraj.jobs --select-tuned`")
        for s in range(1, 6):
            for data in (NG, HD):
                t = tuned[Path(data).name]
                configs = [["--algo", "central"], ["--algo", "fedavg"],
                           ["--algo", "fedprox", "--mu", t["fedprox"]], ["--algo", "fednova"],
                           ["--algo", "fedavgm", "--server-lr", t["fedavgm"]],
                           ["--algo", "fedadam", "--server-lr", t["fedadam"]],
                           ["--algo", "scaffold"], ["--algo", "fedbss"], CLIENT_ONLY, OURS]
                if float(t["fedavg"]) != 1.0:      # FedAvg with a tuned server step, next to plain FedAvg
                    configs.append(["--algo", "fedavg", "--server-lr", t["fedavg"]])
                for e in configs:
                    yield data, "natural", s, e
    elif stage == "ablation":
        for s in range(1, 6):
            for data in (NG, HD):
                for e in (FIXED, SAMPLE_ONLY, OURS + ["--budget", "inverse"], OURS + ["--route", "random"]):
                    yield data, "natural", s, e
        for s in range(1, 4):
            for e in (OURS + ["--agg", "avg"], OURS + ["--tail-pos", "late"], OURS + ["--tail-weight", "one"],
                      OURS + ["--precond", "tensor"], OURS + ["--tail-frac", "0.1"], OURS + ["--tail-frac", "0.3"]):
                yield NG, "natural", s, e
    elif stage == "robust":
        # settings with the strongest client heterogeneity, where the client-level budget should matter most
        for s in range(1, 4):
            for data in (NG, HD):
                for e in (["--algo", "fedavg"], ["--algo", "fednova"], ["--algo", "fedbss"], CLIENT_ONLY, OURS,
                          FIXED):
                    yield data, "motion", s, e
                for e in (["--algo", "fedavg"], ["--algo", "fednova"], CLIENT_ONLY, OURS, FIXED):
                    yield data, "natural", s, e + ["--client-frac", "0.5"]
            for e in (["--algo", "fedavg"], ["--algo", "fednova"], ["--algo", "fedbss"], CLIENT_ONLY, OURS, FIXED):
                yield HD, "recording", s, e + ["--client-frac", "0.1", "--rounds", "100"]
    elif stage == "ssbr-pilot":
        for data, part in ssbr_settings(NG, HD):
            for e in ssbr_configs(SSBR_MU):
                yield data, part, 1, e
    elif stage == "ssbr-main":
        if tuned is None:
            raise SystemExit("ssbr-main needs runs_fedtraj/tuned_ssbr.json: run the ssbr-pilot stage, then "
                             "`python -m fedtraj.jobs --ssbr-select`")
        for s in range(1, 6):
            for data, part in ssbr_settings(NG, HD):
                key = setting_key(data, part)
                if key not in tuned:
                    raise SystemExit(f"tuned_ssbr.json has no entry for {key}: rerun --ssbr-select")
                for e in ssbr_configs([tuned[key]["fedprox"]]):
                    yield data, part, s, e
        for s in SSBR_KEY_SEEDS:           # extra seeds for the key comparisons (decided before the main results)
            for data, part in ssbr_settings(NG, HD):
                for e in ssbr_key_configs(tuned[setting_key(data, part)]["fedprox"]):
                    yield data, part, s, e
    elif stage == "ssbr-robust":
        for s in range(1, 4):
            for e in (["--algo", "fedavg"], CQS, CQS_RANDOM, UDR, SSBR):
                yield NG, "motion", s, e
            for e in (["--algo", "fedavg"], ["--algo", "fednova"], CQS, UDR, SSBR):
                yield HD, "recording", s, e + ["--client-frac", "0.1", "--rounds", "100"]
    elif stage == "ssbr-extra":
        if tuned is None:
            raise SystemExit("ssbr-extra needs runs_fedtraj/tuned_ssbr.json: run `python -m fedtraj.jobs --ssbr-select`")
        for data, part in ssbr_settings(NG, HD):
            beta = tuned.get(setting_key(data, part), {}).get("static_beta")
            if beta is None:
                raise SystemExit(f"tuned_ssbr.json has no static_beta for {setting_key(data, part)}: "
                                 "rerun `python -m fedtraj.jobs --ssbr-select` with the current code")
            for s in range(1, 6):          # static weights with the concentration of UDR, FedBSS-mean, UDR-FS
                for e in (CQS + ["--reweight", "static", "--static-beta", beta],
                          ["--algo", "fedbss", "--bss-split", "mean"], CQS + ["--reweight", "udr-fs"]):
                    yield data, part, s, e
            for s in range(1, 4):          # reversed admission order; half of the clients per round
                yield data, part, s, CQS + ["--cqs-order", "anti"]
                for e in (["--algo", "fedavg"], CQS, UDR, SSBR):
                    yield data, part, s, e + ["--client-frac", "0.5"]
            for e in (["--cqs-floor", "0.4"], ["--cqs-floor", "0.8"], ["--cqs-warmup", "0"], ["--cqs-warmup", "10"],
                      ["--udr-temp", "0.5"], ["--udr-temp", "2"], ["--udr-min", "0.7", "--udr-max", "1.3"],
                      ["--udr-min", "0.3", "--udr-max", "1.7"]):
                yield data, part, 1, SSBR + e      # single-seed sensitivity around Fed-SSBR
            for r in (2, 3):                       # same-seed replicates: run-to-run variation of the scan
                yield data, part, 1, SSBR + ["--run-id", f"cqs-udr-rep{r}_s1"]
        for s in range(1, 4):                      # pacing shape at E = 5 (both give 4.0 n_i visits per round)
            for pace in ("linear", "cosine"):
                yield NG, "natural", s, CQS + ["--local-epochs", "5", "--cqs-pace", pace]
    elif stage == "ssbr-combo":
        if tuned is None:
            raise SystemExit("ssbr-combo needs runs_fedtraj/tuned_ssbr.json: run `python -m fedtraj.jobs --ssbr-select`")
        for s in range(1, 6):                      # seed-major, so that partial results cover both settings
            for data, part in ssbr_settings(NG, HD):
                mu = tuned[setting_key(data, part)]["fedprox"]
                for e in (CQS + ["--nova"], CQS + ["--prox-mu", mu]):
                    yield data, part, s, e
    elif stage == "r17-smoke":
        tiny = ["--rounds", "2", "--local-epochs", "3", "--max-windows", "1500", "--save-checkpoint", "0",
                "--cqs-warmup", "1", "--bss-warmup", "1"]
        for data, part in ssbr_settings(NG, HD):
            for e in (CQS + BN, CQS_DESC + BN, CQS_DESC, CQS, ["--algo", "fedavg"] + LSTM,
                      CQS + LSTM, ["--algo", "fedbss"] + LSTM, CQS_RANDOM + LSTM):
                yield data, part, 1, e + tiny
    elif stage == "r17-order":
        for s in range(1, 9):                      # seed-major, so that partial results cover both settings
            for data, part in ssbr_settings(NG, HD):
                for e in (CQS + BN, CQS_DESC + BN):
                    yield data, part, s, e
        for s in range(1, 4):
            for data, part in ssbr_settings(NG, HD):
                for e in (CQS_DESC, CQS):
                    yield data, part, s, e
    elif stage == "r17-lstm-pilot":
        for data, part in ssbr_settings(NG, HD):
            for lr in R17_LSTM_LR:
                yield data, part, 1, ["--algo", "fedavg", "--lr", lr] + LSTM
    elif stage in ("r17-lstm", "r17-main"):
        if stage == "r17-main":
            yield from specs("r17-order", NG, HD, tuned)
        if tuned is None:
            if stage == "r17-lstm":
                raise SystemExit("r17-lstm needs runs_r17/tuned_r17.json: run the r17-lstm-pilot stage, then "
                                 "`python -m fedtraj.jobs --r17-select`")
            print("r17-main: runs_r17/tuned_r17.json is missing, so only the r17-order runs are listed")
            return
        for s in range(1, 6):
            for data, part in ssbr_settings(NG, HD):
                lr = tuned[setting_key(data, part)]["lstm_lr"]
                for e in (["--algo", "fedavg"], CQS, ["--algo", "fedbss"], CQS_RANDOM):
                    yield data, part, s, e + ["--lr", lr] + LSTM
    else:
        raise SystemExit(f"unknown stage {stage}")


def build_jobs(stage, out_dir=None, ngsim="data/fedtraj_ngsim", highd="data/fedtraj_highd", tuned=None,
               skip_done=True):
    """List of jobs {argv, run_dir, run_id, name}; ``argv`` are the arguments of ``fedtraj.train``."""
    from fedtraj.train import parse_args

    out_dir = out_dir or STAGE_OUT.get(stage, "runs_fedtraj")
    if stage == "main" and tuned is None:
        tuned = load_tuned("runs_fedtraj")
    if stage in ("ssbr-main", "ssbr-extra", "ssbr-combo") and tuned is None:
        tuned = load_tuned(out_dir, "tuned_ssbr.json")
    if stage in ("r17-lstm", "r17-main") and tuned is None:
        tuned = load_tuned(out_dir, "tuned_r17.json")
    jobs, seen = [], set()
    for data, part, seed, extra in specs(stage, ngsim, highd, tuned):
        argv = ["--data-dir", data, "--partition", part, "--seed", str(seed), "--out-dir", out_dir] + extra
        a = parse_args(argv)
        run_dir = Path(out_dir) / Path(data).name / part / a.run_id
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if skip_done and (run_dir / "result.json").exists():
            continue
        jobs.append({"argv": argv, "run_dir": str(run_dir), "run_id": a.run_id,
                     "name": f"{Path(data).name}_{part}_{a.run_id}"})
    return jobs


def load_tuned(out_dir, name="tuned.json"):
    p = Path(out_dir) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def select_tuned(out_dir="runs_fedtraj", ngsim="data/fedtraj_ngsim", highd="data/fedtraj_highd"):
    """Pick, per dataset, the tune-grid value with the lowest best validation 5 s RMSE (seed 1)."""
    from fedtraj.train import parse_args

    tuned = {}
    for data in (ngsim, highd):
        name = Path(data).name
        tuned[name] = {}
        for algo, (flag, values) in TUNE_GRID.items():
            scores = {}
            for v in values:
                a = parse_args(["--data-dir", data, "--partition", "natural", "--seed", "1", "--algo", algo, flag, v])
                f = Path(out_dir) / name / "natural" / a.run_id / "result.json"
                if f.exists():
                    r = json.loads(f.read_text(encoding="utf-8"))
                    scores[v] = float("inf") if r.get("diverged_round") else r["best_val_5s"]
            missing = [v for v in values if v not in scores]
            if missing:
                raise SystemExit(f"{name} {algo}: tune runs missing for {flag} {missing}")
            best = min(scores, key=scores.get)
            tuned[name][algo] = best
            print(f"{name:15s} {algo:8s} {flag} " + "  ".join(f"{v}: {s:.4f}" for v, s in scores.items())
                  + f"  -> {best}")
    p = Path(out_dir) / "tuned.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return tuned


def keff_of(beta, counts):
    """Effective number of clients 1 / sum alpha_i^2 of the static weights alpha_i ~ n_i^beta."""
    w = [float(n) ** beta for n in counts]
    s = sum(w)
    return 1.0 / sum((x / s) ** 2 for x in w)


def match_beta(target, counts):
    """beta >= 0 with K_eff(n^beta) = target; K_eff falls from K (beta = 0) as beta grows, so bisect."""
    if target >= keff_of(0.0, counts):
        return 0.0
    lo, hi = 0.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if keff_of(mid, counts) > target else (lo, mid)
    return 0.5 * (lo + hi)


def ssbr_select(out_dir="runs_fedtraj", ngsim="data/fedtraj_ngsim", highd="data/fedtraj_highd"):
    """After ssbr-pilot (seed 1, validation only): the go / stop check and the FedProx coefficient.

    Gate: continue with ssbr-main only if, in at least one main setting, CQS has a mean validation 5 s RMSE over
    the last 10 rounds at least ``SSBR_GATE_M`` below FedAvg.  FedProx: the grid value with the lowest best
    validation 5 s RMSE, written to tuned_ssbr.json.
    """
    from fedtraj.train import parse_args

    def result(data, part, extra):
        a = parse_args(["--data-dir", data, "--partition", part, "--seed", "1"] + extra)
        f = Path(out_dir) / Path(data).name / part / a.run_id / "result.json"
        if not f.exists():
            raise SystemExit(f"missing {f}: finish the ssbr-pilot stage first")
        return json.loads(f.read_text(encoding="utf-8"))

    def last10(r):
        curve = [c["5s"] for c in r["val_curve"]][-10:]
        return sum(curve) / len(curve)

    def mean_keff(data, part, extra):
        a = parse_args(["--data-dir", data, "--partition", part, "--seed", "1"] + extra)
        f = Path(out_dir) / Path(data).name / part / a.run_id / "log.csv"
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
        vals = [json.loads(r["extra"])["keff"] for r in rows if "keff" in json.loads(r["extra"])]
        if not vals:
            raise SystemExit(f"{f} has no K_eff record")
        return sum(vals) / len(vals)

    tuned, go = {}, []
    print(f"validation 5 s RMSE, seed 1 (m); gate: CQS - FedAvg <= -{SSBR_GATE_M} on the last-10-round mean "
          f"in at least one setting")
    for data, part in ssbr_settings(ngsim, highd):
        key = setting_key(data, part)
        avg, cqs = result(data, part, ["--algo", "fedavg"]), result(data, part, CQS)
        d10, dbest = last10(cqs) - last10(avg), cqs["best_val_5s"] - avg["best_val_5s"]
        go.append(d10 <= -SSBR_GATE_M + 1e-9)        # tolerance: 4.48 - 4.50 is -0.01999... in floating point
        print(f"  {key:22s} FedAvg last-10 {last10(avg):.4f} best {avg['best_val_5s']:.4f} | "
              f"CQS last-10 {last10(cqs):.4f} best {cqs['best_val_5s']:.4f} | "
              f"CQS - FedAvg: last-10 {d10:+.4f}, best {dbest:+.4f}  -> {'passes' if go[-1] else 'does not pass'}")
        scores = {}
        for mu in SSBR_MU:
            r = result(data, part, ["--algo", "fedprox", "--mu", mu])
            scores[mu] = float("inf") if r.get("diverged_round") else r["best_val_5s"]
        best = min(scores, key=scores.get)
        tuned[key] = {"fedprox": best}
        print(f"  {'':22s} FedProx best validation 5 s: " + "  ".join(f"mu {m}: {s:.4f}" for m, s in scores.items())
              + f"  -> mu {best}")
        # static-weight reference of ssbr-extra: n_i^beta with the mean K_eff of Fed-SSBR over all pilot rounds
        counts = result(data, part, SSBR)["train_counts"]
        target = mean_keff(data, part, SSBR)
        beta = match_beta(target, counts)
        tuned[key].update(static_beta=f"{beta:.3f}", keff_fed_ssbr=round(target, 4))
        print(f"  {'':22s} Fed-SSBR mean K_eff {target:.3f} (FedAvg {keff_of(1.0, counts):.3f}, uniform "
              f"{len(counts)}) -> static weights n_i^{beta:.3f}")
    p = Path(out_dir) / "tuned_ssbr.json"
    p.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    if any(go):
        print("GO: the gate passes; run the ssbr-main stage after confirming with the advisor")
    else:
        print("STOP: CQS is not 0.02 m below FedAvg in either main setting; do not add seeds")
    return any(go)


def r17_select(out_dir="runs_r17", ngsim="data/fedtraj_ngsim", highd="data/fedtraj_highd"):
    """After r17-lstm-pilot: per main setting, the LSTM learning rate with the lowest best validation 5 s RMSE of
    FedAvg (seed 1), written to tuned_r17.json.  The file is not written if, in a setting, the selected FedAvg
    model is not below CV extrapolation on validation (the backbone would then not be a meaningful check).
    Returns 0 when the file was written, 2 when pilot runs are missing, 3 when the CV check fails."""
    from fedtraj.train import parse_args

    tuned, ok = {}, True
    for data, part in ssbr_settings(ngsim, highd):
        key = setting_key(data, part)
        scores, cv = {}, None
        for lr in R17_LSTM_LR:
            a = parse_args(["--data-dir", data, "--partition", part, "--seed", "1", "--algo", "fedavg", "--lr", lr]
                           + LSTM)
            f = Path(out_dir) / Path(data).name / part / a.run_id / "result.json"
            if not f.exists():
                print(f"missing {f}: finish the r17-lstm-pilot stage first")
                return 2
            r = json.loads(f.read_text(encoding="utf-8"))
            scores[lr] = float("inf") if r.get("diverged_round") else r["best_val_5s"]
            cv = r["cv_reference"]["valid"]["5s"]
        best = min(scores, key=scores.get)
        below = scores[best] < cv
        ok = ok and below
        tuned[key] = {"lstm_lr": best, "fedavg_best_val_5s": scores, "cv_val_5s": cv}
        print(f"{key:22s} FedAvg-LSTM best validation 5 s: " + "  ".join(f"lr {k}: {v:.4f}" for k, v in scores.items())
              + f"  -> lr {best}; CV {cv:.4f} ({'below CV' if below else 'NOT below CV'})")
    if not ok:
        print("STOP: the LSTM backbone is not below CV extrapolation in every setting; tuned_r17.json not written")
        return 3
    p = Path(out_dir) / "tuned_r17.json"
    p.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--select-tuned", action="store_true")
    ap.add_argument("--ssbr-select", action="store_true",
                    help="after ssbr-pilot: go / stop check and FedProx mu -> tuned_ssbr.json")
    ap.add_argument("--r17-select", action="store_true",
                    help="after r17-lstm-pilot: LSTM learning rate by the FedAvg validation error -> tuned_r17.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ngsim", default="data/fedtraj_ngsim")
    ap.add_argument("--highd", default="data/fedtraj_highd")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()
    if args.select_tuned:
        select_tuned(args.out_dir or "runs_fedtraj", args.ngsim, args.highd)
        return
    if args.ssbr_select:
        ssbr_select(args.out_dir or "runs_fedtraj", args.ngsim, args.highd)
        return
    if args.r17_select:
        raise SystemExit(r17_select(args.out_dir or "runs_r17", args.ngsim, args.highd))
    if not args.stage:
        ap.error("--stage, --select-tuned, --ssbr-select or --r17-select is required")
    for j in build_jobs(args.stage, args.out_dir, args.ngsim, args.highd):
        print(f"{args.python} -m fedtraj.train {' '.join(j['argv'])}")


if __name__ == "__main__":
    main()
