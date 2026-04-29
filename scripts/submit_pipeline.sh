#!/usr/bin/env bash
set -euo pipefail

RUN_IDS=(
  # "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
#   "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
)

AGGREGATION_MODES=(
  # "plain"
  "secure"
)

SELECTION_ID="${SELECTION_ID:-test__max-20__seed-42}"
PERSONA="${PERSONA:-lay}"
SBATCH_SCRIPT="scripts/recommender_pipeline.sbatch"

for run_id in "${RUN_IDS[@]}"; do
  for aggregation_mode in "${AGGREGATION_MODES[@]}"; do
    sbatch "$SBATCH_SCRIPT" "$run_id" "$aggregation_mode" "$SELECTION_ID" "$PERSONA"
  done
done
