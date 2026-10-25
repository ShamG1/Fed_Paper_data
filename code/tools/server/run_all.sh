#!/bin/bash
# Fed-SSBR on one RTX 4090: smoke -> timing (GPU alone) -> ssbr-pilot (packed) -> go / stop check.
# Packing: at most 4 jobs on the GPU, within 22.5 GB by per-job estimates measured on this server
# (training 3.1 GB NGSIM / 4.3 GB highD, plus headroom for scoring and cuDNN workspaces).
cd /data/fedtraj_proj
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
export PYTHONUNBUFFERED=1
PACK="--mem-gb 22.5 --per-gpu 4 --job-gb fedtraj_ngsim=4.5,fedtraj_highd=6.0"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "smoke";      $PY -m fedtraj.launch --stage smoke $PACK > launch_smoke.log 2>&1;      log "smoke exit $?"
log "timing";     $PY -m fedtraj.launch --stage timing --per-gpu 1 > launch_timing.log 2>&1; log "timing exit $?"
$PY -m fedtraj.collect --runs runs_timing --timing > timing_summary.txt 2>&1
log "ssbr-pilot"; $PY -m fedtraj.launch --stage ssbr-pilot $PACK > launch_ssbr_pilot.log 2>&1; log "ssbr-pilot exit $?"
$PY -m fedtraj.jobs --ssbr-select > ssbr_select.txt 2>&1;                                  log "ssbr-select exit $?"
$PY -m fedtraj.collect --runs runs_fedtraj --ref fedavg > pilot_vs_fedavg.txt 2>&1
$PY -m fedtraj.collect --runs runs_fedtraj --ref cqs > pilot_vs_cqs.txt 2>&1
log "all done; send back ssbr_select.txt, pilot_vs_fedavg.txt, pilot_vs_cqs.txt, timing_summary.txt"
