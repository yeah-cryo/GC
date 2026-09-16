#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p outputs/resnet50_critic
"${PYTHON:-.venv/bin/python}" -u tools/train_critic.py "$@" 2>&1 | tee -a outputs/resnet50_critic/train.log
