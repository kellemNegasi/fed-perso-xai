#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_pipeline.sh [clustered] [CLUSTERING_K] [CLUSTERING_WARMUP_ROUNDS] [CLUSTERING_FREEZE_PCA_AFTER_WARMUP] [TOP_K] [CLUSTERING_ENABLE_PCA]

Behavior:
  Default: submit both plain and secure runs for every RUN_ID.
  clustered: submit one clustered run for every RUN_ID.

Environment variables:
  TRAIN_ROUNDS=10
  TRAIN_EPOCHS=5
  TRAIN_BATCH_SIZE=2048
  CLUSTERING_K=3
  CLUSTERING_WARMUP_ROUNDS=0
  CLUSTERING_FREEZE_PCA_AFTER_WARMUP=0
  CLUSTERING_ENABLE_PCA=1
  TOP_K=1,3,5
USAGE
}

RUN_IDS=(
  "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
  "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
)

MODE_ARG="${1:-}"
CLUSTERING_K_ARG="${2:-}"
CLUSTERING_WARMUP_ROUNDS_ARG="${3:-}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG="${4:-}"
TOP_K_ARG="${5:-}"
CLUSTERING_ENABLE_PCA_ARG="${6:-}"

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
if [[ -n "$CLUSTERING_WARMUP_ROUNDS_ARG" && ! "$CLUSTERING_WARMUP_ROUNDS_ARG" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_WARMUP_ROUNDS must be a non-negative integer." >&2
  exit 2
fi
if [[ -n "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG" && ! "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_FREEZE_PCA_AFTER_WARMUP must be 0 or 1." >&2
  exit 2
fi
if [[ -n "$CLUSTERING_ENABLE_PCA_ARG" && ! "$CLUSTERING_ENABLE_PCA_ARG" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_ENABLE_PCA must be 0 or 1." >&2
  exit 2
fi

SELECTION_ID="${SELECTION_ID:-test__max-20__seed-42}"
PERSONA="${PERSONA:-lay}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-100}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
CLUSTERING_K="${CLUSTERING_K_ARG:-${CLUSTERING_K:-3}}"
CLUSTERING_WARMUP_ROUNDS="${CLUSTERING_WARMUP_ROUNDS_ARG:-${CLUSTERING_WARMUP_ROUNDS:-15}}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP="${CLUSTERING_FREEZE_PCA_AFTER_WARMUP_ARG:-${CLUSTERING_FREEZE_PCA_AFTER_WARMUP:-1}}"
TOP_K="${TOP_K_ARG:-${TOP_K:-1,3,5,8}}"
CLUSTERING_ENABLE_PCA="${CLUSTERING_ENABLE_PCA_ARG:-${CLUSTERING_ENABLE_PCA:-0}}"
if [[ ! "$CLUSTERING_K" =~ ^[0-9]+$ ]] || [[ "$CLUSTERING_K" -lt 1 ]]; then
  echo "ERROR: CLUSTERING_K must be a positive integer." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_WARMUP_ROUNDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLUSTERING_WARMUP_ROUNDS must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_FREEZE_PCA_AFTER_WARMUP must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$CLUSTERING_ENABLE_PCA" =~ ^(0|1)$ ]]; then
  echo "ERROR: CLUSTERING_ENABLE_PCA must be 0 or 1." >&2
  exit 2
fi
if [[ -z "$TOP_K" ]]; then
  echo "ERROR: TOP_K must be a non-empty comma-separated list." >&2
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
      "$CLUSTERING_K" \
      "$CLUSTERING_WARMUP_ROUNDS" \
      "$CLUSTERING_FREEZE_PCA_AFTER_WARMUP" \
      "$TOP_K" \
      "$CLUSTERING_ENABLE_PCA"
  done
done
