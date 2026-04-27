#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_recommender_pipeline.sh RUN_ID SELECTION_ID [PERSONA]

Environment variables:
  PYTHON=python                         Python executable to use.
  CLIENTS=all                           Comma-separated client ids or all.
  CONTEXT_FILENAME=candidate_context.parquet
                                        Context file to label/train on.
  LABEL_FILENAME=pairwise_labels.parquet
                                        Label artifact name expected by training/eval.
  SIMULATOR=dirichlet_persona           Labeling simulator name.
  LABEL_SEED=1729                       RNG seed for simulated pairwise labels.
  PERSONA_SEED=42                       RNG seed for persona weight sampling.
  TRAIN_ROUNDS=10                       Federated recommender rounds.
  TRAIN_EPOCHS=5                        Local recommender epochs.
  TRAIN_BATCH_SIZE=64                   Local recommender batch size.
  TRAIN_LEARNING_RATE=0.05              Local recommender learning rate.
  TRAIN_L2_REGULARIZATION=0.0           Local recommender L2 regularization.
  TRAIN_SEED=42                         Federated recommender training seed.
  FIT_FRACTION=1.0                      Fraction of clients sampled for fit.
  EVALUATE_FRACTION=1.0                 Fraction of clients sampled for eval.
  MIN_AVAILABLE_CLIENTS=2               Minimum clients for fit/eval.
  SIMULATION_BACKEND=auto               auto, ray, debug-sequential, sequential_fallback.
  DEBUG_FALLBACK_ON_ERROR=0             Pass --debug-fallback-on-error to training.
  SKIP_LABELING=0                       Skip label-recommender-context when set to 1.
  SECURE_AGGREGATION=0                  Pass --secure-aggregation to training when set to 1.
  SECURE_NUM_HELPERS=5                  Secure aggregation helper count.
  SECURE_PRIVACY_THRESHOLD=2            Secure aggregation privacy threshold.
  SECURE_RECONSTRUCTION_THRESHOLD=      Optional secure aggregation reconstruction threshold.
  SECURE_FIELD_MODULUS=2147483647       Secure aggregation field modulus.
  SECURE_QUANTIZATION_SCALE=65536       Secure aggregation quantization scale.
  SECURE_SEED=0                         Secure aggregation RNG seed.
  TOP_K=1,3,5                           Comma-separated precision@k cutoffs.
  FORCE_TRAINING=0                      Pass --force to train-recommender-federated.
  EVAL_OUTPUT=                          Optional path for evaluate-recommender JSON output.

Pipeline:
  1. label-recommender-context
  2. train-recommender-federated
  3. evaluate-recommender
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

RUN_ID="$1"
SELECTION_ID="$2"
PERSONA="${3:-${PERSONA:-lay}}"

CLIENTS="${CLIENTS:-all}"
CONTEXT_FILENAME="${CONTEXT_FILENAME:-candidate_context.parquet}"
LABEL_FILENAME="${LABEL_FILENAME:-pairwise_labels.parquet}"
SIMULATOR="${SIMULATOR:-dirichlet_persona}"
LABEL_SEED="${LABEL_SEED:-1729}"
PERSONA_SEED="${PERSONA_SEED:-42}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-10}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.05}"
TRAIN_L2_REGULARIZATION="${TRAIN_L2_REGULARIZATION:-0.0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
FIT_FRACTION="${FIT_FRACTION:-1.0}"
EVALUATE_FRACTION="${EVALUATE_FRACTION:-1.0}"
MIN_AVAILABLE_CLIENTS="${MIN_AVAILABLE_CLIENTS:-2}"
SIMULATION_BACKEND="${SIMULATION_BACKEND:-auto}"
DEBUG_FALLBACK_ON_ERROR="${DEBUG_FALLBACK_ON_ERROR:-0}"
SKIP_LABELING="${SKIP_LABELING:-1}"
SECURE_AGGREGATION="${SECURE_AGGREGATION:-0}"
SECURE_NUM_HELPERS="${SECURE_NUM_HELPERS:-5}"
SECURE_PRIVACY_THRESHOLD="${SECURE_PRIVACY_THRESHOLD:-2}"
SECURE_RECONSTRUCTION_THRESHOLD="${SECURE_RECONSTRUCTION_THRESHOLD:-}"
SECURE_FIELD_MODULUS="${SECURE_FIELD_MODULUS:-2147483647}"
SECURE_QUANTIZATION_SCALE="${SECURE_QUANTIZATION_SCALE:-65536}"
SECURE_SEED="${SECURE_SEED:-0}"
TOP_K="${TOP_K:-1,3,5}"
FORCE_TRAINING="${FORCE_TRAINING:-0}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}"

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

TRAIN_EXTRA=()
if [[ "$DEBUG_FALLBACK_ON_ERROR" == "1" ]]; then
  TRAIN_EXTRA+=(--debug-fallback-on-error)
fi
if [[ "$FORCE_TRAINING" == "1" ]]; then
  TRAIN_EXTRA+=(--force)
fi
if [[ "$SECURE_AGGREGATION" == "1" ]]; then
  TRAIN_EXTRA+=(--secure-aggregation)
fi
TRAIN_EXTRA+=(--secure-num-helpers "$SECURE_NUM_HELPERS")
TRAIN_EXTRA+=(--secure-privacy-threshold "$SECURE_PRIVACY_THRESHOLD")
if [[ -n "$SECURE_RECONSTRUCTION_THRESHOLD" ]]; then
  TRAIN_EXTRA+=(--secure-reconstruction-threshold "$SECURE_RECONSTRUCTION_THRESHOLD")
fi
TRAIN_EXTRA+=(--secure-field-modulus "$SECURE_FIELD_MODULUS")
TRAIN_EXTRA+=(--secure-quantization-scale "$SECURE_QUANTIZATION_SCALE")
TRAIN_EXTRA+=(--secure-seed "$SECURE_SEED")

EVAL_EXTRA=()
if [[ -n "$EVAL_OUTPUT" ]]; then
  EVAL_EXTRA+=(--output "$EVAL_OUTPUT")
fi

if [[ "$SKIP_LABELING" == "1" ]]; then
  echo "==> Skipping recommender labeling"
else
  echo "==> Labeling recommender context"
  "$PYTHON" -m fed_perso_xai label-recommender-context \
    --run-id "$RUN_ID" \
    --selection "$SELECTION_ID" \
    --persona "$PERSONA" \
    --simulator "$SIMULATOR" \
    --clients "$CLIENTS" \
    --context-filename "$CONTEXT_FILENAME" \
    --label-filename "$LABEL_FILENAME" \
    --seed "$PERSONA_SEED" \
    --label-seed "$LABEL_SEED"
fi

echo "==> Training federated recommender"
"$PYTHON" -m fed_perso_xai train-recommender-federated \
  --run-id "$RUN_ID" \
  --selection "$SELECTION_ID" \
  --persona "$PERSONA" \
  --clients "$CLIENTS" \
  --context-filename "$CONTEXT_FILENAME" \
  --label-filename "$LABEL_FILENAME" \
  --rounds "$TRAIN_ROUNDS" \
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --learning-rate "$TRAIN_LEARNING_RATE" \
  --l2-regularization "$TRAIN_L2_REGULARIZATION" \
  --seed "$TRAIN_SEED" \
  --fit-fraction "$FIT_FRACTION" \
  --evaluate-fraction "$EVALUATE_FRACTION" \
  --min-available-clients "$MIN_AVAILABLE_CLIENTS" \
  --simulation-backend "$SIMULATION_BACKEND" \
  --top-k "$TOP_K" \
  "${TRAIN_EXTRA[@]}"

echo "==> Evaluating federated recommender"
"$PYTHON" -m fed_perso_xai evaluate-recommender \
  --run-id "$RUN_ID" \
  --selection "$SELECTION_ID" \
  --persona "$PERSONA" \
  --clients "$CLIENTS" \
  --context-filename "$CONTEXT_FILENAME" \
  --label-filename "$LABEL_FILENAME" \
  --top-k "$TOP_K" \
  "${EVAL_EXTRA[@]}"

echo "==> Recommender pipeline complete"
echo "Run ID: $RUN_ID"
echo "Selection: $SELECTION_ID"
echo "Persona: $PERSONA"
