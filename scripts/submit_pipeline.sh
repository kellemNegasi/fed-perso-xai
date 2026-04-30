#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_pipeline.sh [clustered] [CLUSTERING_K]

Behavior:
  Default: submit both plain and secure runs for every RUN_ID.
  clustered: submit one clustered run for every RUN_ID.

Environment variables:
  TRAIN_ROUNDS=10
  TRAIN_EPOCHS=5
  TRAIN_BATCH_SIZE=2048
  CLUSTERING_K=3
USAGE
}

RUN_IDS=(
  "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
  "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
)

MODE_ARG="${1:-}"
CLUSTERING_K_ARG="${2:-}"

case "${MODE_ARG,,}" in
  "")
    SUBMISSION_MODES=(
      "plain"
      "secure"
    )
    ;;
  clustered|--clustered)
    SUBMISSION_MODES=("clustered")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unsupported mode '$MODE_ARG'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -n "$CLUSTERING_K_ARG" && ! "$CLUSTERING_K_ARG" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_K must be a positive integer." >&2
  exit 2
fi

SELECTION_ID="${SELECTION_ID:-test__max-20__seed-42}"
PERSONA="${PERSONA:-lay}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-15}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
CLUSTERING_K="${CLUSTERING_K_ARG:-${CLUSTERING_K:-2}}"
if [[ ! "$CLUSTERING_K" =~ ^[0-9]+$ ]] || [[ "$CLUSTERING_K" -lt 1 ]]; then
  echo "ERROR: CLUSTERING_K must be a positive integer." >&2
  exit 2
fi
SBATCH_SCRIPT="scripts/recommender_pipeline.sbatch"

for run_id in "${RUN_IDS[@]}"; do
  for submission_mode in "${SUBMISSION_MODES[@]}"; do
    sbatch \
      "$SBATCH_SCRIPT" \
      "$run_id" \
      "$submission_mode" \
      "$SELECTION_ID" \
      "$PERSONA" \
      "$TRAIN_ROUNDS" \
      "$TRAIN_EPOCHS" \
      "$TRAIN_BATCH_SIZE" \
      "$CLUSTERING_K"
  done
done
