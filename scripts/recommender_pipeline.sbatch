#!/bin/bash
#
# Generic single-job Slurm script for rerunning one recommender pipeline.
# Submit with:
#   sbatch scripts/recommender_pipeline.sbatch
#
# To reuse this script for another task, change only the TARGET_CMD block below.
#

# Job metadata -----------------------------------------------------------------
#SBATCH --job-name=recommender-run
#SBATCH --output=logs/slurm-%x.%j.out
#SBATCH --error=logs/slurm-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=15
#SBATCH --mem=60G
#SBATCH --time=4:00:00

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

# Workload ---------------------------------------------------------------------
echo "[$(date --iso-8601=seconds)] Starting recommender pipeline job"
echo "  project_root=${PROJECT_ROOT}"
echo "  host=${HOSTNAME}"
echo "  pwd=$(pwd)"

# Change only this block when reusing the script for another target.
# With the current recommender runtime, each Ray client reserves 1 CPU.
# Setting cpus-per-task to the number of clients allows all clients to run concurrently.
# Set SKIP_LABELING=1 here when pairwise labels already exist and should be reused.
# Secure aggregation and clustered training are toggled here via environment variables
# passed to the wrapper.
TARGET_CMD=(
  env
  FORCE_TRAINING=1
  SKIP_LABELING=1
  CONTEXT_FILENAME=candidate_context.parquet
  RECOMMENDER_TYPE=svm_rank
  TRAIN_BATCH_SIZE=2048
  SIMULATION_BACKEND=auto
  SECURE_AGGREGATION=1
  CLUSTERED=0
  SECURE_NUM_HELPERS=10
  SECURE_PRIVACY_THRESHOLD=2
  SECURE_RECONSTRUCTION_THRESHOLD=3
  SECURE_FIELD_MODULUS=2147483647
  SECURE_QUANTIZATION_SCALE=65536
  SECURE_SEED=7
  scripts/run_recommender_pipeline.sh
  federated-training-adult_income-20260427t063710289874+0000-logistic_regression-15clients-alpha0.3-seed42-536f2cd41ed2
  test__max-20__seed-42
  lay
)

printf '  command='
printf '%q ' "${TARGET_CMD[@]}"
printf '\n'

"${TARGET_CMD[@]}"

echo "[$(date --iso-8601=seconds)] Recommender pipeline job completed"
