#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
exec .venv/bin/python -u tools/visualize_genimage_tsne.py \
  --config configs/experiments/resnet50_online40k_genimage_tsne.yaml "$@"
