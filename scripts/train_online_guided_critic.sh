#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec "${PYTHON:-.venv/bin/python}" -u tools/train_online_guided_critic.py "$@"
