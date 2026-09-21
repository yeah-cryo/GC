#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
config="configs/experiments/safe_sd14_genimage.yaml"
output=""
for ((index=1; index<=$#; index++)); do
  if [[ "${!index}" == "--config" ]]; then
    next=$((index + 1))
    config="${!next}"
  elif [[ "${!index}" == "--output" ]]; then
    next=$((index + 1))
    output="${!next}"
  fi
done
if [[ -z "$output" ]]; then
  output=$("${PYTHON:-.venv/bin/python}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output"])' "$config")
fi
mkdir -p "$output"
"${PYTHON:-.venv/bin/python}" -u tools/train_safe_genimage.py "$@" 2>&1 | tee -a "$output/train.log"
