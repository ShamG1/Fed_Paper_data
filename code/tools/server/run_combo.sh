#!/bin/bash
# Added 2026-10-24, after the main results: CQS + FedNova and CQS + FedProx (stage ssbr-combo, 20 runs).
# Waits for run_rest.sh (ssbr-robust, ssbr-extra) to finish, then runs the stage on the one RTX 4090 and rewrites
# results_vs_fedavg.txt and results_vs_cqs.txt.  Finished runs are skipped, so it can be restarted.
#     setsid nohup ./run_combo.sh >> run_all.log 2>&1 < /dev/null &
cd /data/fedtraj_proj
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
export PYTHONUNBUFFERED=1
PACK="--mem-gb 22.5 --per-gpu 4 --job-gb fedtraj_ngsim=4.5,fedtraj_highd=6.0"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if pgrep -f run_rest.sh > /dev/null; then
  log "run_combo.sh queued: ssbr-combo starts when run_rest.sh (ssbr-robust, ssbr-extra) has finished"
  while pgrep -f run_rest.sh > /dev/null; do sleep 60; done
fi
log "ssbr-combo"
$PY -m fedtraj.launch --stage ssbr-combo $PACK > launch_ssbr_combo.log 2>&1
log "ssbr-combo exit $?"
$PY -m fedtraj.collect --runs runs_fedtraj --ref fedavg > results_vs_fedavg.txt 2>&1
$PY -m fedtraj.collect --runs runs_fedtraj --ref cqs > results_vs_cqs.txt 2>&1
log "all stages done (incl. ssbr-combo); results_vs_fedavg.txt and results_vs_cqs.txt rewritten"
