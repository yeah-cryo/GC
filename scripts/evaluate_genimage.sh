#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p outputs/resnet50_critic/genimage_evaluation
"${PYTHON:-.venv/bin/python}" -u tools/evaluate_genimage.py "$@" 2>&1 | tee -a outputs/resnet50_critic/genimage_evaluation/evaluate.log
