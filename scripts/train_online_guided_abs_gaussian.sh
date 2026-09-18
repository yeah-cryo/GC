#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/train_online_guided_critic.sh \
  --config configs/experiments/resnet50_online_guided_abs_gaussian_coco20k.yaml "$@"
