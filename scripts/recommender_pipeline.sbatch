#!/bin/bash
#
# Generic single-job Slurm script for rerunning one recommender pipeline.
# Submit with:
#   sbatch scripts/recommender_pipeline.sbatch RUN_ID MODE [SELECTION_ID] [PERSONA] [TRAIN_ROUNDS] [TRAIN_EPOCHS] [TRAIN_BATCH_SIZE] [SKIP_LABELING] [CLUSTERING_K] [CLUSTERING_WARMUP_ROUNDS] [CLUSTERING_FREEZE_PCA_AFTER_WARMUP] [TOP_K] [CLUSTERING_ENABLE_PCA]
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
# Environment setup ------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/helios/home/kelemnegasi/fed-xai/fed-perso-xai}"
cd "$PROJECT_ROOT"

# module load python/3.14.0
# Load the latest available Python 3.14 module, if any
module load $(module -t avail python 2>&1 | grep '^python/3\.14' | head -n 1)
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Check module/venv setup." >&2
  exit 1
fi

usage() {
  echo "Usage: sbatch scripts/recommender_pipeline.sbatch RUN_ID MODE [SELECTION_ID] [PERSONA] [TRAIN_ROUNDS] [TRAIN_EPOCHS] [TRAIN_BATCH_SIZE] [SKIP_LABELING] [CLUSTERING_K] [CLUSTERING_WARMUP_ROUNDS] [CLUSTERING_FREEZE_PCA_AFTER_WARMUP] [TOP_K] [CLUSTERING_ENABLE_PCA]" >&2
  echo "  RUN_ID: predictive training run id" >&2
  echo "  MODE: plain, secure, clustered, 1/0, or true/false" >&2
  echo "  SELECTION_ID defaults to test__max-20__seed-42" >&2
  echo "  PERSONA defaults to lay" >&2
  echo "  TRAIN_ROUNDS defaults to 10" >&2
  echo "  TRAIN_EPOCHS defaults to 5" >&2
  echo "  TRAIN_BATCH_SIZE defaults to 2048" >&2
  echo "  SKIP_LABELING defaults to 1" >&2
  echo "  CLUSTERING_K defaults to 3" >&2
  echo "  CLUSTERING_WARMUP_ROUNDS defaults to 0" >&2
  echo "  CLUSTERING_FREEZE_PCA_AFTER_WARMUP defaults to 0" >&2
  echo "  CLUSTERING_ENABLE_PCA defaults to 1" >&2
  echo "  TOP_K defaults to 1,3,5" >&2
}

if [[ $# -lt 2 || $# -gt 13 ]]; then
  usage
  exit 2
fi

RUN_ID="$1"
MODE_RAW="$2"
SELECTION_ID="${3:-${SELECTION_ID:-test__max-20__seed-42}}"
PERSONA="${4:-${PERSONA:-lay}}"
TRAIN_ROUNDS="${5:-${TRAIN_ROUNDS:-10}}"
TRAIN_EPOCHS="${6:-${TRAIN_EPOCHS:-5}}"
TRAIN_BATCH_SIZE="${7:-${TRAIN_BATCH_SIZE:-2048}}"
SKIP_LABELING="${8:-${SKIP_LABELING:-1}}"
CLUSTERING_K="${9:-${CLUSTERING_K:-3}}"
CLUSTERING_WARMUP_ROUNDS="${10:-${CLUSTERING_WARMUP_ROUNDS:-0}}"
CLUSTERING_FREEZE_PCA_AFTER_WARMUP="${11:-${CLUSTERING_FREEZE_PCA_AFTER_WARMUP:-0}}"
TOP_K="${12:-${TOP_K:-1,3,5}}"
CLUSTERING_ENABLE_PCA="${13:-${CLUSTERING_ENABLE_PCA:-1}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
CLIENT_NUM_CPUS="${CLIENT_NUM_CPUS:-1}"

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
if [[ ! "$SKIP_LABELING" =~ ^(0|1)$ ]]; then
  echo "ERROR: SKIP_LABELING must be 0 or 1." >&2
  exit 2
fi
if [[ -z "$TOP_K" ]]; then
  echo "ERROR: TOP_K must be a non-empty comma-separated list." >&2
  exit 2
fi

case "${MODE_RAW,,}" in
  1|true|secure)
    SECURE_AGGREGATION_VALUE=1
    CLUSTERED_VALUE=0
    MODE_VALUE="secure"
    ;;
  0|false|plain)
    SECURE_AGGREGATION_VALUE=0
    CLUSTERED_VALUE=0
    MODE_VALUE="plain"
    ;;
  clustered)
    SECURE_AGGREGATION_VALUE=0
    CLUSTERED_VALUE=1
    MODE_VALUE="clustered"
    ;;
  *)
    echo "ERROR: MODE must be one of plain, secure, clustered, 1, 0, true, or false." >&2
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
echo "  mode=${MODE_VALUE}"
echo "  secure_aggregation=${SECURE_AGGREGATION_VALUE}"
echo "  clustered=${CLUSTERED_VALUE}"
echo "  train_rounds=${TRAIN_ROUNDS}"
echo "  train_epochs=${TRAIN_EPOCHS}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  skip_labeling=${SKIP_LABELING}"
echo "  clustering_k=${CLUSTERING_K}"
echo "  clustering_warmup_rounds=${CLUSTERING_WARMUP_ROUNDS}"
echo "  clustering_freeze_pca_after_warmup=${CLUSTERING_FREEZE_PCA_AFTER_WARMUP}"
echo "  clustering_enable_pca=${CLUSTERING_ENABLE_PCA}"
echo "  top_k=${TOP_K}"
echo "  ray_num_cpus=${RAY_NUM_CPUS}"
echo "  client_num_cpus=${CLIENT_NUM_CPUS}"

TARGET_CMD=(
  env
  FORCE_TRAINING=1
  SKIP_LABELING=${SKIP_LABELING}
  CONTEXT_FILENAME=candidate_context.parquet
  RECOMMENDER_TYPE=svm_rank
  TRAIN_ROUNDS=${TRAIN_ROUNDS}
  TRAIN_EPOCHS=${TRAIN_EPOCHS}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
  SIMULATION_BACKEND=auto
  RAY_NUM_CPUS=${RAY_NUM_CPUS}
  CLIENT_NUM_CPUS=${CLIENT_NUM_CPUS}
  SECURE_AGGREGATION=${SECURE_AGGREGATION_VALUE}
  CLUSTERED=${CLUSTERED_VALUE}
  CLUSTERING_K=${CLUSTERING_K}
  CLUSTERING_WARMUP_ROUNDS=${CLUSTERING_WARMUP_ROUNDS}
  CLUSTERING_FREEZE_PCA_AFTER_WARMUP=${CLUSTERING_FREEZE_PCA_AFTER_WARMUP}
  CLUSTERING_ENABLE_PCA=${CLUSTERING_ENABLE_PCA}
  TOP_K=${TOP_K}
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
