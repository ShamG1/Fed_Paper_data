#!/bin/bash
# Pilot extension (2026-10-22): FedProx mu = 1e-1 and 1 on seed 1 of both main settings (4 runs; the 24 finished
# pilot runs are skipped), then the gate / mu / static-beta selection and the summary tables again.
#     setsid nohup ./run_pilot_ext.sh >> run_all.log 2>&1 < /dev/null &
cd /data/fedtraj_proj
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
export PYTHONUNBUFFERED=1
PACK="--mem-gb 22.5 --per-gpu 4 --job-gb fedtraj_ngsim=4.5,fedtraj_highd=6.0"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "ssbr-pilot extension: FedProx mu 1e-1, 1"
$PY -m fedtraj.launch --stage ssbr-pilot $PACK > launch_ssbr_pilot_ext.log 2>&1
log "ssbr-pilot extension exit $?"
$PY -m fedtraj.jobs --ssbr-select > ssbr_select.txt 2>&1
log "ssbr-select exit $?"
$PY -m fedtraj.collect --runs runs_fedtraj --ref fedavg > pilot_vs_fedavg.txt 2>&1
$PY -m fedtraj.collect --runs runs_fedtraj --ref cqs > pilot_vs_cqs.txt 2>&1
log "pilot extension done; ssbr_select.txt updated"
