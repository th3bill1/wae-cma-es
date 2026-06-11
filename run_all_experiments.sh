#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Python interpreter not found. Set PYTHON=/path/to/python and retry." >&2
    exit 1
fi

mkdir -p results/raw results/summary results/plots

echo "Running CMA-ES experiments..."
"$PYTHON_BIN" run_experiments.py

echo "Generating summaries and plots..."
"$PYTHON_BIN" reproduce_all.py

echo "All experiments, summaries, and plots have been regenerated."
