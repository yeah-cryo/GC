#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/train_saved_guided_critic.sh \
  --config configs/experiments/resnet50_saved_online40k_first10k_cosine.yaml "$@"
