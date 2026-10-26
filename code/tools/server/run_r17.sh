#!/bin/bash
# Checks after R16 on the second server (two RTX 4090, project /root/fedtraj_proj/code, env /root/envs/t24 with
# Python 3.10, torch 2.4.1+cu121, numpy 2.0.1).  Order: r17-smoke (tiny runs, stop on any failure) ->
# r17-lstm-pilot -> --r17-select -> r17-main (r17-order, then r17-lstm with the selected learning rate).
# Finished runs are skipped, so the script can be restarted.
#     setsid nohup ./tools/server/run_r17.sh >> run_r17.log 2>&1 < /dev/null &
cd /root/fedtraj_proj/code || exit 1
PY=/root/envs/t24/bin/python
export PYTHONUNBUFFERED=1
# per-job GPU memory: the values of the first server for the CV-anchored predictor (274 runs, four per 4090, no
# failure); for the LSTM the peak measured on this server with evaluation batches of 1024, plus the CUDA context
MEM="--mem-gb 22.5 --retries 3 --job-gb fedtraj_ngsim=4.5,fedtraj_highd=6.0,lstm:fedtraj_ngsim=3.7,lstm:fedtraj_highd=4.9"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "r17-smoke"
if ! $PY -m fedtraj.launch --stage r17-smoke $MEM --per-gpu 4 > launch_r17_smoke.log 2>&1; then
  log "r17-smoke failed: see launch_r17_smoke.log; stopping"
  exit 1
fi
log "r17-lstm-pilot"
$PY -m fedtraj.launch --stage r17-lstm-pilot $MEM --per-gpu 3 > launch_r17_lstm_pilot.log 2>&1
log "r17-lstm-pilot exit $?"
$PY -m fedtraj.jobs --r17-select > r17_select.txt 2>&1
log "r17-select exit $? (0: rate written; 3: LSTM not below CV, so r17-main runs r17-order only)"
$PY -m fedtraj.launch --stage r17-main $MEM --per-gpu 4 > launch_r17_main.log 2>&1
log "r17-main exit $?"
log "all r17 stages done"
