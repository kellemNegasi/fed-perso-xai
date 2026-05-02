#!/bin/bash
#
# Slurm array script for rerunning recommender-context preparation.
# Submit with:
#   sbatch --array=0-2 scripts/prepare_recommender_context.sbatch RUN_ID_FILE [SELECTION_ID]
#

#SBATCH --job-name=prep-rec-ctx
#SBATCH --output=logs/slurm-%x.%A_%a.out
#SBATCH --error=logs/slurm-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/helios/home/kelemnegasi/fed-xai/fed-perso-xai}"
cd "$PROJECT_ROOT"

module load $(module -t avail python 2>&1 | grep '^python/3\.14' | head -n 1)
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Check module/venv setup." >&2
  exit 1
fi

usage() {
  echo "Usage: sbatch --array=0-N scripts/prepare_recommender_context.sbatch RUN_ID_FILE [SELECTION_ID]" >&2
  echo "  RUN_ID_FILE: text file with one run_id per line" >&2
  echo "  SELECTION_ID defaults to test__max-20__seed-42" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

RUN_ID_FILE="$1"
SELECTION_ID="${2:-${SELECTION_ID:-test__max-20__seed-42}}"
EXPLAINERS="${EXPLAINERS:-all}"
CONFIGS="${CONFIGS:-all}"
CLIENTS="${CLIENTS:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$RUN_ID_FILE" ]]; then
  echo "ERROR: RUN_ID_FILE does not exist: $RUN_ID_FILE" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set; submit with sbatch --array." >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
RUN_ID="$(sed -n "$((TASK_ID + 1))p" "$RUN_ID_FILE")"
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: no run_id found for SLURM_ARRAY_TASK_ID=${TASK_ID} in ${RUN_ID_FILE}" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting recommender-context preparation"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  task_id=${TASK_ID}"
echo "  run_id=${RUN_ID}"
echo "  selection_id=${SELECTION_ID}"
echo "  explainers=${EXPLAINERS}"
echo "  configs=${CONFIGS}"
echo "  clients=${CLIENTS}"

"$PYTHON_BIN" -m fed_perso_xai prepare-recommender-context   --run-id "$RUN_ID"   --selection "$SELECTION_ID"   --explainers "$EXPLAINERS"   --configs "$CONFIGS"   --clients "$CLIENTS"

echo "[$(date --iso-8601=seconds)] Recommender-context preparation completed"
