#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_pipeline.sh [clustered]

Behavior:
  Default: submit both plain and secure runs for every RUN_ID.
  clustered: submit one clustered run for every RUN_ID.
USAGE
}

RUN_IDS=(
  "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
  "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
)

MODE_ARG="${1:-}"

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

SELECTION_ID="${SELECTION_ID:-test__max-20__seed-42}"
PERSONA="${PERSONA:-lay}"
SBATCH_SCRIPT="scripts/recommender_pipeline.sbatch"

for run_id in "${RUN_IDS[@]}"; do
  for submission_mode in "${SUBMISSION_MODES[@]}"; do
    sbatch "$SBATCH_SCRIPT" "$run_id" "$submission_mode" "$SELECTION_ID" "$PERSONA"
  done
done
