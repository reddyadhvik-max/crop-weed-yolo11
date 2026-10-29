#!/bin/bash
# Keeps training alive overnight: waits for any running train.py, then (re)runs it until the test eval prints.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 20); do
  while pgrep -f "^\.venv/bin/python train\.py" >/dev/null; do sleep 30; done
  grep -aq "TEST mAP" train.log && { echo "$(date) done" >> supervise.log; exit 0; }
  echo "$(date) train.py not running and no TEST result -> restart #$i" >> supervise.log
  .venv/bin/python train.py >> train.log 2>&1
  sleep 20
done
