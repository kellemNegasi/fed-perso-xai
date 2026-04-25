#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_explain_eval_pipeline.sh [configs/job_launcher.yml]

Environment variables:
  PYTHON=python                         Python executable to use.
  EXECUTION_MODE=slurm                  slurm, local, or skip.
  WAIT_FOR_SLURM=1                      Wait for submitted Slurm array before aggregation.
  ALLOW_PARTIAL=0                       Pass --allow-partial during aggregation.
  FORCE_TRAINING=0                      Pass --force to launch-experiment-jobs.
  LABEL_RECOMMENDER_CONTEXT=1           Run simulated user pairwise labeling after context prep.
  RECOMMENDER_PERSONA=lay               Bundled persona name for label-recommender-context.
  RECOMMENDER_CONTEXT_FILE=candidate_context.parquet
                                        Context file to label; use all_candidate_context.parquet to include dominated candidates.
  RECOMMENDER_SIMULATOR=dirichlet_persona
  RECOMMENDER_SEED=42
  RECOMMENDER_LABEL_SEED=1729

Pipeline:
  1. launch-experiment-jobs: prepare/train/plan and write the .sbatch script
  2. metric computation: submit Slurm array, run locally, or skip
  3. aggregate-explain-eval for every explainer/config pair in the JSONL plan
  4. prepare-recommender-context with --explainers all --configs all
  5. label-recommender-context with simulated pairwise user preferences
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG="${1:-configs/job_launcher.yml}"
EXECUTION_MODE="${EXECUTION_MODE:-slurm}"
WAIT_FOR_SLURM="${WAIT_FOR_SLURM:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
FORCE_TRAINING="${FORCE_TRAINING:-0}"
LABEL_RECOMMENDER_CONTEXT="${LABEL_RECOMMENDER_CONTEXT:-1}"
RECOMMENDER_PERSONA="${RECOMMENDER_PERSONA:-lay}"
RECOMMENDER_CONTEXT_FILE="${RECOMMENDER_CONTEXT_FILE:-candidate_context.parquet}"
RECOMMENDER_SIMULATOR="${RECOMMENDER_SIMULATOR:-dirichlet_persona}"
RECOMMENDER_SEED="${RECOMMENDER_SEED:-42}"
RECOMMENDER_LABEL_SEED="${RECOMMENDER_LABEL_SEED:-1729}"

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

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: launcher config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p job_launcher/pipeline_runs
STAMP="$(date +%Y%m%dT%H%M%S)"
LAUNCH_OUTPUT="job_launcher/pipeline_runs/launch_${STAMP}.json"

echo "==> Launching training and explain/eval planning"
LAUNCH_ARGS=(launch-experiment-jobs --config "$CONFIG")
if [[ "$FORCE_TRAINING" == "1" ]]; then
  LAUNCH_ARGS+=(--force)
fi
"$PYTHON" -m fed_perso_xai "${LAUNCH_ARGS[@]}" | tee "$LAUNCH_OUTPUT"

eval "$(
  "$PYTHON" - "$LAUNCH_OUTPUT" "$CONFIG" <<'PY'
import json
import shlex
import sys
from pathlib import Path

import yaml

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8")) or {}
runs = payload.get("runs") or []
if len(runs) != 1:
    raise SystemExit(f"Expected exactly one launcher run, found {len(runs)}.")

run = runs[0]
plan = run.get("explain_eval_plan") or {}
explain_cfg = config.get("explain_eval") or {}
split = str(explain_cfg.get("split", "test"))
max_instances = int(explain_cfg.get("max_instances", 50))
random_state = int(explain_cfg.get("random_state", 42))
selection = f"{split}__max-{max_instances}__seed-{random_state}"

values = {
    "RUN_ID": run["run_id"],
    "PLAN_PATH": plan["plan_path"],
    "SBATCH_PATH": run.get("slurm_script_path", ""),
    "SELECTION_ID": selection,
    "JOB_COUNT": str(plan.get("job_count", "")),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

echo "==> Run ID: $RUN_ID"
echo "==> Selection: $SELECTION_ID"
echo "==> Plan: $PLAN_PATH"
echo "==> Jobs: $JOB_COUNT"

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "ERROR: plan file was not created: $PLAN_PATH" >&2
  exit 1
fi

case "$EXECUTION_MODE" in
  slurm)
    if [[ -z "$SBATCH_PATH" || ! -f "$SBATCH_PATH" ]]; then
      echo "ERROR: Slurm script was not created. Check explain_eval.slurm.enabled in $CONFIG." >&2
      exit 1
    fi
    echo "==> Submitting Slurm array: $SBATCH_PATH"
    SUBMIT_OUTPUT="$(sbatch "$SBATCH_PATH" "$PLAN_PATH")"
    echo "$SUBMIT_OUTPUT"
    SLURM_JOB_ID="$(echo "$SUBMIT_OUTPUT" | awk '{print $NF}')"
    if [[ "$WAIT_FOR_SLURM" == "1" ]]; then
      echo "==> Waiting for Slurm job $SLURM_JOB_ID"
      while squeue -j "$SLURM_JOB_ID" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$SLURM_JOB_ID" -h)" ]]; do
        sleep 30
      done
    else
      echo "==> Not waiting for Slurm. Re-run with EXECUTION_MODE=skip after jobs finish to aggregate/context."
      exit 0
    fi
    ;;
  local)
    echo "==> Running explain/eval plan rows locally"
    PLAN_ROWS="$(wc -l < "$PLAN_PATH")"
    for ((idx = 0; idx < PLAN_ROWS; idx++)); do
      echo "==> Local plan row $idx / $((PLAN_ROWS - 1))"
      "$PYTHON" -m fed_perso_xai run-explain-eval-plan-item --plan "$PLAN_PATH" --index "$idx"
    done
    ;;
  skip)
    echo "==> Skipping metric computation; assuming plan outputs already exist"
    ;;
  *)
    echo "ERROR: EXECUTION_MODE must be slurm, local, or skip. Got: $EXECUTION_MODE" >&2
    exit 1
    ;;
esac

AGGREGATE_EXTRA=()
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  AGGREGATE_EXTRA+=(--allow-partial)
fi

echo "==> Aggregating every explainer/config pair from the plan"
"$PYTHON" - "$PLAN_PATH" <<'PY' | while IFS=$'\t' read -r EXPLAINER CONFIG_ID; do
import json
import sys
from pathlib import Path

pairs = set()
with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        pairs.add((str(row["explainer"]), str(row["config_id"])))
for explainer, config_id in sorted(pairs):
    print(f"{explainer}\t{config_id}")
PY
  echo "==> Aggregating $EXPLAINER / $CONFIG_ID"
  "$PYTHON" -m fed_perso_xai aggregate-explain-eval \
    --run-id "$RUN_ID" \
    --selection "$SELECTION_ID" \
    --explainer "$EXPLAINER" \
    --config "$CONFIG_ID" \
    "${AGGREGATE_EXTRA[@]}"
done

echo "==> Preparing recommender context"
"$PYTHON" -m fed_perso_xai prepare-recommender-context \
  --run-id "$RUN_ID" \
  --selection "$SELECTION_ID" \
  --explainers all \
  --configs all \
  --clients all

if [[ "$LABEL_RECOMMENDER_CONTEXT" == "1" ]]; then
  echo "==> Labeling recommender context with simulated user preferences"
  "$PYTHON" -m fed_perso_xai label-recommender-context \
    --run-id "$RUN_ID" \
    --selection "$SELECTION_ID" \
    --persona "$RECOMMENDER_PERSONA" \
    --simulator "$RECOMMENDER_SIMULATOR" \
    --clients all \
    --context-filename "$RECOMMENDER_CONTEXT_FILE" \
    --seed "$RECOMMENDER_SEED" \
    --label-seed "$RECOMMENDER_LABEL_SEED"
fi

echo "==> Pipeline complete"
echo "Run ID: $RUN_ID"
echo "Selection: $SELECTION_ID"
