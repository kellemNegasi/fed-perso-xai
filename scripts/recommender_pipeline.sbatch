#!/bin/bash
#
# Generic single-job Slurm script for rerunning one recommender pipeline.
# Submit with:
#   sbatch scripts/recommender_pipeline.sbatch RUN_ID SECURE_AGGREGATION [SELECTION_ID] [PERSONA]
#

# Job metadata -----------------------------------------------------------------
#SBATCH --job-name=recommender-run
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=7
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --nodelist=ares3
# Environment setup ------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/helios/home/kelemnegasi/fed-xai/fed-perso-xai}"
cd "$PROJECT_ROOT"

module load python/3.14.0
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Check module/venv setup." >&2
  exit 1
fi

usage() {
  echo "Usage: sbatch scripts/recommender_pipeline.sbatch RUN_ID SECURE_AGGREGATION [SELECTION_ID] [PERSONA]" >&2
  echo "  RUN_ID: predictive training run id" >&2
  echo "  SECURE_AGGREGATION: 1/0 or secure/plain" >&2
  echo "  SELECTION_ID defaults to test__max-20__seed-42" >&2
  echo "  PERSONA defaults to lay" >&2
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

RUN_ID="$1"
SECURE_AGGREGATION_RAW="$2"
SELECTION_ID="${3:-${SELECTION_ID:-test__max-20__seed-42}}"
PERSONA="${4:-${PERSONA:-lay}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-1}"

case "${SECURE_AGGREGATION_RAW,,}" in
  1|true|secure)
    SECURE_AGGREGATION_VALUE=1
    ;;
  0|false|plain)
    SECURE_AGGREGATION_VALUE=0
    ;;
  *)
    echo "ERROR: SECURE_AGGREGATION must be one of 1, 0, secure, plain, true, false." >&2
    exit 2
    ;;
esac

# Workload ---------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting recommender pipeline job"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"
echo "  run_id=${RUN_ID}"
echo "  selection_id=${SELECTION_ID}"
echo "  persona=${PERSONA}"
echo "  secure_aggregation=${SECURE_AGGREGATION_VALUE}"
echo "  ray_num_cpus=${RAY_NUM_CPUS}"
echo "  client_num_cpus=${CLIENT_NUM_CPUS}"

TARGET_CMD=(
  env
  FORCE_TRAINING=1
  SKIP_LABELING=1
  CONTEXT_FILENAME=candidate_context.parquet
  RECOMMENDER_TYPE=svm_rank
  TRAIN_BATCH_SIZE=2048
  SIMULATION_BACKEND=auto
  RAY_NUM_CPUS=${RAY_NUM_CPUS}
  CLIENT_NUM_CPUS=${CLIENT_NUM_CPUS}
  SECURE_AGGREGATION=${SECURE_AGGREGATION_VALUE}
  CLUSTERED=0
  SECURE_NUM_HELPERS=15
  SECURE_PRIVACY_THRESHOLD=2
  SECURE_RECONSTRUCTION_THRESHOLD=3
  SECURE_FIELD_MODULUS=3037000493
  SECURE_QUANTIZATION_SCALE=16777216
  SECURE_SEED=7
  scripts/run_recommender_pipeline.sh
  "${RUN_ID}"
  "${SELECTION_ID}"
  "${PERSONA}"
)

printf '  command='
printf '%q ' "${TARGET_CMD[@]}"
printf '\n'

"${TARGET_CMD[@]}"

echo "[$(date --iso-8601=seconds)] Recommender pipeline job completed"
