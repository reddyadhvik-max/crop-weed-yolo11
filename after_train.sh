#!/bin/bash
# Waits for training + test-split eval to finish, then runs the accuracy test suite on the GPU.
cd "$(dirname "$0")"
until grep -aq "TEST mAP" train.log && ! pgrep -f "^\.venv/bin/python train\.py" >/dev/null; do sleep 60; done
.venv/bin/python test_model.py 2>&1 | grep -avE "it/s|^\s*$" > test_results.log
