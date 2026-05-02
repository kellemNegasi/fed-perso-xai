#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/prepare_recommender_context.sh [SELECTION_ID]

Environment variables:
  EXPLAINERS=all                       Comma-separated explainer names or all.
  CONFIGS=all                          Comma-separated config ids or all.
  CLIENTS=all                          Comma-separated client ids or all.
  RUN_ID_FILE=job_launcher/plans/prepare_recommender_context_run_ids.txt
                                        File written for the sbatch array, one run_id per line.

Behavior:
  - Writes the configured run ids to RUN_ID_FILE.
  - Submits one Slurm array task per run id via scripts/prepare_recommender_context.sbatch.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

SELECTION_ID="${1:-${SELECTION_ID:-test__max-20__seed-42}}"
EXPLAINERS="${EXPLAINERS:-all}"
CONFIGS="${CONFIGS:-all}"
CLIENTS="${CLIENTS:-all}"
RUN_ID_FILE="${RUN_ID_FILE:-job_launcher/plans/prepare_recommender_context_run_ids.txt}"
SBATCH_SCRIPT="scripts/prepare_recommender_context.sbatch"

RUN_IDS=(
  "federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b"
  "federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2"
  "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
)

mkdir -p "$(dirname "$RUN_ID_FILE")"
printf '%s
' "${RUN_IDS[@]}" > "$RUN_ID_FILE"

ARRAY_SPEC="0-$((${#RUN_IDS[@]} - 1))"

echo "Submitting recommender-context preparation array"
echo "  sbatch_script=${SBATCH_SCRIPT}"
echo "  run_id_file=${RUN_ID_FILE}"
echo "  selection_id=${SELECTION_ID}"
echo "  explainers=${EXPLAINERS}"
echo "  configs=${CONFIGS}"
echo "  clients=${CLIENTS}"
echo "  array=${ARRAY_SPEC}"

sbatch   --array="${ARRAY_SPEC}"   --export=ALL,EXPLAINERS="${EXPLAINERS}",CONFIGS="${CONFIGS}",CLIENTS="${CLIENTS}"   "$SBATCH_SCRIPT"   "$RUN_ID_FILE"   "$SELECTION_ID"
