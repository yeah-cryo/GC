#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p outputs/sd14_soft_prompt_bce_perceptual
"${PYTHON:-.venv/bin/python}" -u tools/train_prompt.py "$@" 2>&1 | tee -a outputs/sd14_soft_prompt_bce_perceptual/train.log
