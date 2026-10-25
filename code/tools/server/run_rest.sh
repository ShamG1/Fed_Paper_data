#!/bin/bash
# Fed-SSBR after the pilot gate.  Start ONLY after the go decision has been confirmed:
#     setsid nohup ./run_rest.sh >> run_all.log 2>&1 < /dev/null &
# Waits for a running pilot extension (run_pilot_ext.sh), so that ssbr-main uses the final FedProx mu.
# Runs ssbr-main -> ssbr-robust -> ssbr-extra on the one RTX 4090 (up to 4 jobs packed by GPU memory),
# then writes results_vs_fedavg.txt and results_vs_cqs.txt.  Finished runs are skipped, so it can be restarted.
cd /data/fedtraj_proj
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
export PYTHONUNBUFFERED=1
PACK="--mem-gb 22.5 --per-gpu 4 --job-gb fedtraj_ngsim=4.5,fedtraj_highd=6.0"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if pgrep -f run_pilot_ext.sh > /dev/null; then
  log "waiting for run_pilot_ext.sh (FedProx mu extension) to finish"
  while pgrep -f run_pilot_ext.sh > /dev/null; do sleep 60; done
fi
if ! grep -q static_beta runs_fedtraj/tuned_ssbr.json 2>/dev/null; then
  log "runs_fedtraj/tuned_ssbr.json has no static_beta: run  $PY -m fedtraj.jobs --ssbr-select  first"
  exit 1
fi
if grep -q "^STOP" ssbr_select.txt 2>/dev/null; then
  log "ssbr_select.txt says STOP: not starting the main study"
  exit 1
fi
if ! grep -q "mu 1:" ssbr_select.txt 2>/dev/null; then
  log "ssbr_select.txt does not cover the extended FedProx grid (mu up to 1): check run_pilot_ext.sh first"
  exit 1
fi
for st in ssbr-main ssbr-robust ssbr-extra; do
  log "$st"
  $PY -m fedtraj.launch --stage "$st" $PACK > "launch_${st//-/_}.log" 2>&1
  log "$st exit $?"
done
$PY -m fedtraj.collect --runs runs_fedtraj --ref fedavg > results_vs_fedavg.txt 2>&1
$PY -m fedtraj.collect --runs runs_fedtraj --ref cqs > results_vs_cqs.txt 2>&1
log "all stages done; send back results_vs_fedavg.txt and results_vs_cqs.txt"
