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
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00

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
TARGET_CMD=(
  env
  FORCE_TRAINING=1
  CONTEXT_FILENAME=candidate_context.parquet
  SIMULATION_BACKEND=auto
  scripts/run_recommender_pipeline.sh
  federated-training-adult_income-20260425t192946577949+0000-logistic_regression-10clients-alpha1.0-seed42-dba03a50b07b
  test__max-20__seed-42
  lay
)

printf '  command='
printf '%q ' "${TARGET_CMD[@]}"
printf '\n'

"${TARGET_CMD[@]}"

echo "[$(date --iso-8601=seconds)] Recommender pipeline job completed"
