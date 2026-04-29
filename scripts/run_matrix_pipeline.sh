#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_matrix_pipeline.sh [BASE_CONFIG] [PERSONA]

Environment variables:
  PYTHON=python                         Python executable to use.
  ALPHAS=0.3,1.0,5,10                  Comma-separated partition alpha values.
  NUM_CLIENTS=10,15                    Comma-separated client counts.
  EXECUTION_MODE=slurm                 Passed through to run_explain_eval_pipeline.sh.
  WAIT_FOR_SLURM=1                     Passed through to run_explain_eval_pipeline.sh.
  ALLOW_PARTIAL=0                      Passed through to run_explain_eval_pipeline.sh.
  FORCE_TRAINING=0                     Passed through to both pipeline scripts.
  PERSONA=lay                          Recommender persona if not given as argv[2].
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
  EVAL_OUTPUT_DIR=                      Optional directory for per-run recommender eval JSON.
  MATRIX_OUTPUT_DIR=job_launcher/matrix_runs
                                        Stores generated configs and logs.

Pipeline per matrix cell:
  1. Generate a one-run launcher YAML from BASE_CONFIG with one alpha and one num_clients.
  2. run_explain_eval_pipeline.sh to do data prep, model training, aggregation, and context prep.
  3. run_recommender_pipeline.sh to do labeling, recommender training, and recommender evaluation.

Notes:
  - This wrapper disables label-recommender-context inside run_explain_eval_pipeline.sh
    to avoid labeling the same context twice.
  - BASE_CONFIG should be a launcher YAML like configs/job_launcher.yml.
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
MATRIX_OUTPUT_DIR="${MATRIX_OUTPUT_DIR:-job_launcher/matrix_runs}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"

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

mkdir -p "$MATRIX_OUTPUT_DIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="$MATRIX_OUTPUT_DIR/$STAMP"
mkdir -p "$RUN_ROOT"

if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
  mkdir -p "$EVAL_OUTPUT_DIR"
fi

mapfile -t ALPHA_VALUES < <("$PYTHON" - <<'PY' "$ALPHAS"
import sys
values = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
for value in values:
    print(value)
PY
)

mapfile -t CLIENT_VALUES < <("$PYTHON" - <<'PY' "$NUM_CLIENTS"
import sys
values = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
for value in values:
    print(value)
PY
)

if [[ "${#ALPHA_VALUES[@]}" -eq 0 ]]; then
  echo "ERROR: ALPHAS must contain at least one value." >&2
  exit 1
fi
if [[ "${#CLIENT_VALUES[@]}" -eq 0 ]]; then
  echo "ERROR: NUM_CLIENTS must contain at least one value." >&2
  exit 1
fi

echo "==> Base config: $BASE_CONFIG"
echo "==> Persona: $PERSONA"
echo "==> Alphas: ${ALPHA_VALUES[*]}"
echo "==> Client counts: ${CLIENT_VALUES[*]}"
echo "==> Output root: $RUN_ROOT"

for alpha in "${ALPHA_VALUES[@]}"; do
  for num_clients in "${CLIENT_VALUES[@]}"; do
    combo_dir="$RUN_ROOT/alpha-${alpha}_clients-${num_clients}"
    mkdir -p "$combo_dir"
    combo_config="$combo_dir/job_launcher.yml"
    explain_log="$combo_dir/explain_eval_pipeline.log"

    echo "==> Generating config for alpha=$alpha num_clients=$num_clients"
    "$PYTHON" - <<'PY' "$BASE_CONFIG" "$combo_config" "$alpha" "$num_clients"
import sys
from pathlib import Path

import yaml

base_config = Path(sys.argv[1])
output_config = Path(sys.argv[2])
alpha = float(sys.argv[3])
num_clients = int(sys.argv[4])

payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
partition = dict(payload.get("partition") or {})
partition["alphas"] = [float(alpha)]
partition["num_clients"] = [int(num_clients)]
payload["partition"] = partition

output_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY

    echo "==> Running explain/eval pipeline for alpha=$alpha num_clients=$num_clients"
    LABEL_RECOMMENDER_CONTEXT=0 \
    PYTHON="$PYTHON" \
    EXECUTION_MODE="${EXECUTION_MODE:-slurm}" \
    WAIT_FOR_SLURM="${WAIT_FOR_SLURM:-1}" \
    ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}" \
    FORCE_TRAINING="${FORCE_TRAINING:-0}" \
    scripts/run_explain_eval_pipeline.sh "$combo_config" | tee "$explain_log"

    eval "$(
      "$PYTHON" - <<'PY' "$explain_log"
import shlex
import sys
from pathlib import Path

run_id = ""
selection_id = ""
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.startswith("Run ID: "):
        run_id = line.split(": ", 1)[1].strip()
    elif line.startswith("Selection: "):
        selection_id = line.split(": ", 1)[1].strip()
if not run_id or not selection_id:
    raise SystemExit("Could not parse Run ID / Selection from explain/eval pipeline log.")
print(f"RUN_ID={shlex.quote(run_id)}")
print(f"SELECTION_ID={shlex.quote(selection_id)}")
PY
    )"

    echo "==> Running recommender pipeline for run_id=$RUN_ID selection=$SELECTION_ID"
    recommender_log="$combo_dir/recommender_pipeline.log"
    recommender_args=("$RUN_ID" "$SELECTION_ID" "$PERSONA")

    if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
      eval_output_path="$EVAL_OUTPUT_DIR/recommender_eval_alpha-${alpha}_clients-${num_clients}.json"
    else
      eval_output_path=""
    fi

    env_args=(
      "PYTHON=$PYTHON"
      "CLIENTS=${CLIENTS:-all}"
      "CONTEXT_FILENAME=${CONTEXT_FILENAME:-candidate_context.parquet}"
      "LABEL_FILENAME=${LABEL_FILENAME:-pairwise_labels.parquet}"
      "SIMULATOR=${SIMULATOR:-dirichlet_persona}"
      "LABEL_SEED=${LABEL_SEED:-1729}"
      "PERSONA_SEED=${PERSONA_SEED:-42}"
      "TRAIN_ROUNDS=${TRAIN_ROUNDS:-10}"
      "TRAIN_EPOCHS=${TRAIN_EPOCHS:-5}"
      "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}"
      "TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE:-0.05}"
      "TRAIN_L2_REGULARIZATION=${TRAIN_L2_REGULARIZATION:-0.0}"
      "TRAIN_SEED=${TRAIN_SEED:-42}"
      "FIT_FRACTION=${FIT_FRACTION:-1.0}"
      "EVALUATE_FRACTION=${EVALUATE_FRACTION:-1.0}"
      "MIN_AVAILABLE_CLIENTS=${MIN_AVAILABLE_CLIENTS:-2}"
      "SIMULATION_BACKEND=${SIMULATION_BACKEND:-auto}"
      "DEBUG_FALLBACK_ON_ERROR=${DEBUG_FALLBACK_ON_ERROR:-0}"
      "TOP_K=${TOP_K:-1,3,5}"
      "FORCE_TRAINING=${FORCE_TRAINING:-0}"
    )
    if [[ -n "$eval_output_path" ]]; then
      env_args+=("EVAL_OUTPUT=$eval_output_path")
    fi

    env "${env_args[@]}" scripts/run_recommender_pipeline.sh "${recommender_args[@]}" | tee "$recommender_log"
  done
done

echo "==> Matrix pipeline complete"
echo "Output root: $RUN_ROOT"
