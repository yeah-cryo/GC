#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
output_dir=$("${PYTHON:-.venv/bin/python}" -c 'import argparse, yaml; p=argparse.ArgumentParser(add_help=False); p.add_argument("--config", default="configs/experiments/sd14_classifier_guidance.yaml"); a,_=p.parse_known_args(); print(yaml.safe_load(open(a.config))["output"])' "$@")
mkdir -p "$output_dir"
"${PYTHON:-.venv/bin/python}" -u tools/sample_classifier_guidance.py "$@" 2>&1 | tee -a "$output_dir/sample.log"
