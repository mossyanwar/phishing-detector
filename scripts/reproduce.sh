#!/usr/bin/env bash
# Rebuild the dataset, retrain both models and regenerate every result file
# and figure. Expects the raw data from scripts/download_data.py and the venv
# described in README.md. Does not call any LLM API.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY scripts/prepare_data.py
$PY scripts/train_baseline.py --split split
$PY scripts/train_baseline.py --split time
$PY scripts/train_embedding.py
$PY scripts/evaluate.py
$PY scripts/leakage_check.py
$PY scripts/top_features.py
$PY scripts/robustness.py
$PY scripts/error_analysis.py
$PY scripts/latency.py
$PY scripts/run_llm_eval.py --make-sample
$PY scripts/run_llm_eval.py --compare-baselines
