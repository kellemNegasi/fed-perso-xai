#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/submit_matrix_pipeline.sh [BASE_CONFIG] [PERSONA]

Environment variables:
  PYTHON=python                         Python executable to use.
  ALPHAS=0.3,1.0,5,10                  Comma-separated partition alpha values.
  NUM_CLIENTS=10,15                    Comma-separated client counts.
  ARRAY_CONCURRENCY=                   Optional Slurm array concurrency cap.
  MATRIX_OUTPUT_DIR=job_launcher/matrix_runs
                                        Passed through to run_matrix_pipeline.sh.
  EVAL_OUTPUT_DIR=                      Optional directory for recommender eval JSON.
  FORCE_TRAINING=0                     Passed through to pipeline scripts.
  EXECUTION_MODE=slurm                 Passed through to run_matrix_pipeline.sh.
  WAIT_FOR_SLURM=1                     Passed through to run_matrix_pipeline.sh.
  ALLOW_PARTIAL=0                      Passed through to run_matrix_pipeline.sh.
  CLIENTS=all                          Passed through to run_recommender_pipeline.sh.
  CONTEXT_FILENAME=candidate_context.parquet
  LABEL_FILENAME=pairwise_labels.parquet
  SIMULATOR=dirichlet_persona
  LABEL_SEED=1729
  PERSONA_SEED=42
  TRAIN_ROUNDS=10
  TRAIN_EPOCHS=5
  TRAIN_BATCH_SIZE=64
  TRAIN_LEARNING_RATE=0.05
  TRAIN_L2_REGULARIZATION=0.0
  TRAIN_SEED=42
  FIT_FRACTION=1.0
  EVALUATE_FRACTION=1.0
  MIN_AVAILABLE_CLIENTS=2
  SIMULATION_BACKEND=auto
  DEBUG_FALLBACK_ON_ERROR=0
  TOP_K=1,3,5
  SLURM_JOB_NAME=matrix-pipeline       Slurm job name.
  SLURM_LOG_DIR=logs                   Slurm stdout/stderr directory.
  SLURM_MODULE_LOAD=python/3.14.0      Optional module load line.
  SLURM_VENV_PATH=.venv/bin/activate   Optional venv activation path.
  SLURM_SBATCH_ARGS=                   Extra sbatch args, split on newlines.

Pipeline:
  - Submit one Slurm array task per (alpha, num_clients) combination.
  - Each array task runs scripts/run_matrix_pipeline.sh with one alpha and one client count.

Notes:
  - scripts/run_matrix_pipeline.sh already drives the full pipeline for one matrix cell:
    data prep, aggregation, model training, recommender training, and evaluation.
  - This wrapper only handles Slurm submission and task-to-combination mapping.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BASE_CONFIG="${1:-configs/job_launcher.yml}"
PERSONA="${2:-${PERSONA:-lay}}"

ALPHAS="${ALPHAS:-0.3,1.0,5,10}"
NUM_CLIENTS="${NUM_CLIENTS:-10,15}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    echo "ERROR: no Python executable found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

eval "$(
  "$PYTHON" - <<'PY' "$BASE_CONFIG"
import json
import shlex
import sys
from pathlib import Path

import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
slurm_cfg = (((payload.get("explain_eval") or {}).get("slurm")) or {})

def shell(name: str, value: str) -> str:
    return f"{name}={shlex.quote(str(value))}"

sbatch_args = [str(arg).strip() for arg in (slurm_cfg.get("sbatch_args") or []) if str(arg).strip()]
print(shell("CONFIG_SLURM_JOB_NAME", slurm_cfg.get("job_name", "matrix-pipeline")))
print(shell("CONFIG_SLURM_LOG_DIR", slurm_cfg.get("log_dir", "logs")))
print(shell("CONFIG_SLURM_MODULE_LOAD", slurm_cfg.get("module_load", "python/3.14.0")))
print(shell("CONFIG_SLURM_VENV_PATH", slurm_cfg.get("venv_path", ".venv/bin/activate")))
print(shell("CONFIG_SLURM_SBATCH_ARGS_JSON", json.dumps(sbatch_args)))
PY
)"

SLURM_JOB_NAME="${SLURM_JOB_NAME:-${CONFIG_SLURM_JOB_NAME:-matrix-pipeline}}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${CONFIG_SLURM_LOG_DIR:-logs}}"
SLURM_MODULE_LOAD="${SLURM_MODULE_LOAD:-${CONFIG_SLURM_MODULE_LOAD:-python/3.14.0}}"
SLURM_VENV_PATH="${SLURM_VENV_PATH:-${CONFIG_SLURM_VENV_PATH:-.venv/bin/activate}}"

mkdir -p "$SLURM_LOG_DIR"
mkdir -p job_launcher/slurm

COMBO_FILE="$(mktemp job_launcher/slurm/matrix_combos.XXXXXX.tsv)"
SBATCH_SCRIPT="$(mktemp job_launcher/slurm/submit_matrix_pipeline.XXXXXX.sbatch)"

"$PYTHON" - <<'PY' "$ALPHAS" "$NUM_CLIENTS" "$COMBO_FILE"
import itertools
import sys
from pathlib import Path

alphas = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
num_clients = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
if not alphas:
    raise SystemExit("ALPHAS must contain at least one value.")
if not num_clients:
    raise SystemExit("NUM_CLIENTS must contain at least one value.")
rows = ["{}\t{}".format(alpha, client_count) for alpha, client_count in itertools.product(alphas, num_clients)]
Path(sys.argv[3]).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

COMBO_COUNT="$(wc -l < "$COMBO_FILE")"
if (( COMBO_COUNT <= 0 )); then
  echo "ERROR: no matrix combinations were generated." >&2
  exit 1
fi

ARRAY_SPEC="0-$((COMBO_COUNT - 1))"
if [[ -n "$ARRAY_CONCURRENCY" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_CONCURRENCY}"
fi

PROJECT_ROOT="$(pwd)"
SBATCH_ARGS=()
if [[ -n "${SLURM_SBATCH_ARGS:-}" ]]; then
  while IFS= read -r line; do
    if [[ -n "${line// }" ]]; then
      SBATCH_ARGS+=("$line")
    fi
  done <<< "${SLURM_SBATCH_ARGS}"
else
  mapfile -t SBATCH_ARGS < <(
    "$PYTHON" - <<'PY' "${CONFIG_SLURM_SBATCH_ARGS_JSON:-[]}"
import json
import sys

for arg in json.loads(sys.argv[1]):
    text = str(arg).strip()
    if text:
        print(text)
PY
  )
fi

if [[ "${#SBATCH_ARGS[@]}" -eq 0 ]]; then
  SBATCH_ARGS+=("--time=02:00:00")
fi

{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=${SLURM_JOB_NAME}"
  echo "#SBATCH --array=${ARRAY_SPEC}"
  echo "#SBATCH --output=${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_%A_%a.out"
  echo "#SBATCH --error=${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_%A_%a.err"
  for arg in "${SBATCH_ARGS[@]}"; do
    echo "#SBATCH ${arg}"
  done
  cat <<'SCRIPT'

set -euo pipefail

PROJECT_ROOT="__PROJECT_ROOT__"
cd "$PROJECT_ROOT"
SCRIPT
  if [[ -n "$SLURM_MODULE_LOAD" ]]; then
    echo "module load ${SLURM_MODULE_LOAD}"
  fi
  if [[ -n "$SLURM_VENV_PATH" ]]; then
    cat <<SCRIPT
if [[ -f "${SLURM_VENV_PATH}" ]]; then
  source "${SLURM_VENV_PATH}"
fi
SCRIPT
  fi
  cat <<'SCRIPT'

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set." >&2
  exit 2
fi

COMBO_FILE="__COMBO_FILE__"
if [[ ! -f "$COMBO_FILE" ]]; then
  echo "ERROR: combo file not found: $COMBO_FILE" >&2
  exit 2
fi

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMBO_FILE")"
if [[ -z "$ROW" ]]; then
  echo "ERROR: no matrix row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=$'\t' read -r ALPHA NUM_CLIENTS <<< "$ROW"

echo "[$(date --iso-8601=seconds)] Starting matrix task"
echo "  alpha=${ALPHA}"
echo "  num_clients=${NUM_CLIENTS}"
echo "  base_config=__BASE_CONFIG__"
echo "  persona=__PERSONA__"
echo "  host=${HOSTNAME}"

env \
  PYTHON="__PYTHON__" \
  ALPHAS="${ALPHA}" \
  NUM_CLIENTS="${NUM_CLIENTS}" \
  MATRIX_OUTPUT_DIR="__MATRIX_OUTPUT_DIR__" \
  EVAL_OUTPUT_DIR="__EVAL_OUTPUT_DIR__" \
  FORCE_TRAINING="__FORCE_TRAINING__" \
  EXECUTION_MODE="__EXECUTION_MODE__" \
  WAIT_FOR_SLURM="__WAIT_FOR_SLURM__" \
  ALLOW_PARTIAL="__ALLOW_PARTIAL__" \
  CLIENTS="__CLIENTS__" \
  CONTEXT_FILENAME="__CONTEXT_FILENAME__" \
  LABEL_FILENAME="__LABEL_FILENAME__" \
  SIMULATOR="__SIMULATOR__" \
  LABEL_SEED="__LABEL_SEED__" \
  PERSONA_SEED="__PERSONA_SEED__" \
  TRAIN_ROUNDS="__TRAIN_ROUNDS__" \
  TRAIN_EPOCHS="__TRAIN_EPOCHS__" \
  TRAIN_BATCH_SIZE="__TRAIN_BATCH_SIZE__" \
  TRAIN_LEARNING_RATE="__TRAIN_LEARNING_RATE__" \
  TRAIN_L2_REGULARIZATION="__TRAIN_L2_REGULARIZATION__" \
  TRAIN_SEED="__TRAIN_SEED__" \
  FIT_FRACTION="__FIT_FRACTION__" \
  EVALUATE_FRACTION="__EVALUATE_FRACTION__" \
  MIN_AVAILABLE_CLIENTS="__MIN_AVAILABLE_CLIENTS__" \
  SIMULATION_BACKEND="__SIMULATION_BACKEND__" \
  DEBUG_FALLBACK_ON_ERROR="__DEBUG_FALLBACK_ON_ERROR__" \
  TOP_K="__TOP_K__" \
  scripts/run_matrix_pipeline.sh "__BASE_CONFIG__" "__PERSONA__"

echo "[$(date --iso-8601=seconds)] Completed matrix task"
SCRIPT
} > "$SBATCH_SCRIPT"

"$PYTHON" - <<'PY' \
  "$SBATCH_SCRIPT" \
  "$PROJECT_ROOT" \
  "$COMBO_FILE" \
  "$BASE_CONFIG" \
  "$PERSONA" \
  "$PYTHON" \
  "${MATRIX_OUTPUT_DIR:-job_launcher/matrix_runs}" \
  "${EVAL_OUTPUT_DIR:-}" \
  "${FORCE_TRAINING:-0}" \
  "${EXECUTION_MODE:-slurm}" \
  "${WAIT_FOR_SLURM:-1}" \
  "${ALLOW_PARTIAL:-0}" \
  "${CLIENTS:-all}" \
  "${CONTEXT_FILENAME:-candidate_context.parquet}" \
  "${LABEL_FILENAME:-pairwise_labels.parquet}" \
  "${SIMULATOR:-dirichlet_persona}" \
  "${LABEL_SEED:-1729}" \
  "${PERSONA_SEED:-42}" \
  "${TRAIN_ROUNDS:-10}" \
  "${TRAIN_EPOCHS:-5}" \
  "${TRAIN_BATCH_SIZE:-64}" \
  "${TRAIN_LEARNING_RATE:-0.05}" \
  "${TRAIN_L2_REGULARIZATION:-0.0}" \
  "${TRAIN_SEED:-42}" \
  "${FIT_FRACTION:-1.0}" \
  "${EVALUATE_FRACTION:-1.0}" \
  "${MIN_AVAILABLE_CLIENTS:-2}" \
  "${SIMULATION_BACKEND:-auto}" \
  "${DEBUG_FALLBACK_ON_ERROR:-0}" \
  "${TOP_K:-1,3,5}"
import sys
from pathlib import Path

(
    script_path,
    project_root,
    combo_file,
    base_config,
    persona,
    python_exec,
    matrix_output_dir,
    eval_output_dir,
    force_training,
    execution_mode,
    wait_for_slurm,
    allow_partial,
    clients,
    context_filename,
    label_filename,
    simulator,
    label_seed,
    persona_seed,
    train_rounds,
    train_epochs,
    train_batch_size,
    train_learning_rate,
    train_l2_regularization,
    train_seed,
    fit_fraction,
    evaluate_fraction,
    min_available_clients,
    simulation_backend,
    debug_fallback_on_error,
    top_k,
) = sys.argv[1:]

replacements = {
    "__PROJECT_ROOT__": project_root,
    "__COMBO_FILE__": combo_file,
    "__BASE_CONFIG__": base_config,
    "__PERSONA__": persona,
    "__PYTHON__": python_exec,
    "__MATRIX_OUTPUT_DIR__": matrix_output_dir,
    "__EVAL_OUTPUT_DIR__": eval_output_dir,
    "__FORCE_TRAINING__": force_training,
    "__EXECUTION_MODE__": execution_mode,
    "__WAIT_FOR_SLURM__": wait_for_slurm,
    "__ALLOW_PARTIAL__": allow_partial,
    "__CLIENTS__": clients,
    "__CONTEXT_FILENAME__": context_filename,
    "__LABEL_FILENAME__": label_filename,
    "__SIMULATOR__": simulator,
    "__LABEL_SEED__": label_seed,
    "__PERSONA_SEED__": persona_seed,
    "__TRAIN_ROUNDS__": train_rounds,
    "__TRAIN_EPOCHS__": train_epochs,
    "__TRAIN_BATCH_SIZE__": train_batch_size,
    "__TRAIN_LEARNING_RATE__": train_learning_rate,
    "__TRAIN_L2_REGULARIZATION__": train_l2_regularization,
    "__TRAIN_SEED__": train_seed,
    "__FIT_FRACTION__": fit_fraction,
    "__EVALUATE_FRACTION__": evaluate_fraction,
    "__MIN_AVAILABLE_CLIENTS__": min_available_clients,
    "__SIMULATION_BACKEND__": simulation_backend,
    "__DEBUG_FALLBACK_ON_ERROR__": debug_fallback_on_error,
    "__TOP_K__": top_k,
}

path = Path(script_path)
text = path.read_text(encoding="utf-8")
for key, value in replacements.items():
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
PY

chmod +x "$SBATCH_SCRIPT"

echo "==> Matrix combo file: $COMBO_FILE"
echo "==> Slurm script: $SBATCH_SCRIPT"
echo "==> Submitting array: $ARRAY_SPEC"
SUBMIT_OUTPUT="$(sbatch "$SBATCH_SCRIPT")"
echo "$SUBMIT_OUTPUT"
