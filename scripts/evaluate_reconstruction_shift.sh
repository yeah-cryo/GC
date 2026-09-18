#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
exec .venv/bin/python -u tools/evaluate_reconstruction_shift.py \
  --config configs/experiments/reconstruction_shift_online40k_real100.yaml "$@"
