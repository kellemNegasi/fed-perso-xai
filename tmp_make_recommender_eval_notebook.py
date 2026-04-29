import json
from pathlib import Path


cells = []


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


md(
    """# Recommender Held-Out Evaluation Report

This notebook loads the final recommender evaluation for a single `run_id` and summarizes the **held-out test split**.

It is designed for the new recommender flow where:
- train/test are split by `dataset_index`
- final metrics come from the saved `evaluation_summary.json`
- missing recommender artifacts raise a clear error instead of silently falling back
"""
)

code(
    """from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
"""
)

code(
    """# Edit these values to inspect another recommender run.
RUN_ID = "federated-training-adult_income-20260426t223642651433+0000-logistic_regression-10clients-alpha0.3-seed42-e8df09baaba3"
SELECTION_ID = "test__max-20__seed-42"
PERSONA = "lay"
RECOMMENDER_MODEL_KEY = "pairwise_logistic_fedavg"
"""
)

code(
    """def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root containing pyproject.toml")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_id(run_id: str) -> dict[str, object]:
    info: dict[str, object] = {"run_id": run_id}
    match = re.search(
        r"federated-training-(?P<dataset>.+?)-(?P<timestamp>\\d{8}t\\d{6}\\d+\\+\\d+)-(?P<model>.+?)-(?P<num_clients>\\d+)clients-alpha(?P<alpha>[^-]+)-seed(?P<seed>\\d+)",
        run_id,
    )
    if not match:
        return info
    info.update(match.groupdict())
    info["num_clients"] = int(info["num_clients"])
    info["seed"] = int(info["seed"])
    try:
        info["alpha"] = float(info["alpha"])
    except ValueError:
        pass
    return info


def resolve_train_dir(repo_root: Path, run_id: str, selection_id: str, persona: str, model_key: str) -> Path:
    return (
        repo_root
        / "federated"
        / "runs"
        / run_id
        / "recommender_training"
        / selection_id
        / persona
        / model_key
    )


def require_recommender_artifacts(train_dir: Path, run_id: str) -> dict[str, Path]:
    paths = {
        "evaluation_summary": train_dir / "evaluation_summary.json",
        "training_metadata": train_dir / "training_metadata.json",
        "model_artifact": train_dir / "model" / "global_recommender.npz",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Run {run_id!r} does not have completed recommender artifacts in {train_dir}. Missing: {missing_text}."
        )
    return paths


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def make_metric_card(title: str, value: object, subtitle: str = "") -> str:
    subtitle_html = f"<div style='font-size:12px;color:#5f6368'>{subtitle}</div>" if subtitle else ""
    return (
        "<div style='padding:14px 16px;border:1px solid #d0d7de;border-radius:12px;"
        "background:#f8fafc;min-width:180px'>"
        f"<div style='font-size:12px;text-transform:uppercase;color:#5f6368;letter-spacing:0.04em'>{title}</div>"
        f"<div style='font-size:26px;font-weight:700;margin-top:4px'>{format_value(value)}</div>"
        f"{subtitle_html}"
        "</div>"
    )


def build_title_block(run_info: dict[str, object], training_metadata: dict, selection_id: str, persona: str) -> HTML:
    config = training_metadata.get("config", {})
    top_k = config.get("top_k", [])
    cards = [
        make_metric_card("Dataset", run_info.get("dataset")),
        make_metric_card("Model", run_info.get("model")),
        make_metric_card("Alpha", run_info.get("alpha")),
        make_metric_card("Clients", run_info.get("num_clients")),
        make_metric_card("K", top_k, "precision@K reported"),
        make_metric_card("Persona", persona),
        make_metric_card("Selection", selection_id),
        make_metric_card("Split", "held-out test", "final evaluation"),
    ]
    html = (
        "<div style='margin:8px 0 18px 0'>"
        "<div style='font-size:28px;font-weight:700;margin-bottom:6px'>Recommender Held-Out Test Results</div>"
        f"<div style='font-size:14px;color:#5f6368;margin-bottom:14px'><code>{run_info.get('run_id')}</code></div>"
        "<div style='display:flex;flex-wrap:wrap;gap:10px'>"
        + "".join(cards)
        + "</div></div>"
    )
    return HTML(html)


def safe_metric_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in ["pearson", "precision_at_1", "precision_at_3", "precision_at_5"]:
        if column in frame.columns and frame[column].notna().any():
            columns.append(column)
    return columns


def summarize_metric(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    valid = frame[["client_id", metric]].dropna()
    if valid.empty:
        return {
            "metric": metric,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "best_client": None,
            "worst_client": None,
        }
    best_idx = valid[metric].idxmax()
    worst_idx = valid[metric].idxmin()
    return {
        "metric": metric,
        "mean": valid[metric].mean(),
        "std": valid[metric].std(ddof=0),
        "min": valid[metric].min(),
        "max": valid[metric].max(),
        "range": valid[metric].max() - valid[metric].min(),
        "best_client": frame.loc[best_idx, "client_id"],
        "worst_client": frame.loc[worst_idx, "client_id"],
    }
"""
)

code(
    """REPO_ROOT = find_repo_root()
TRAIN_DIR = resolve_train_dir(REPO_ROOT, RUN_ID, SELECTION_ID, PERSONA, RECOMMENDER_MODEL_KEY)
ARTIFACTS = require_recommender_artifacts(TRAIN_DIR, RUN_ID)

training_metadata = load_json(ARTIFACTS["training_metadata"])
evaluation_summary = load_json(ARTIFACTS["evaluation_summary"])
run_info = parse_run_id(RUN_ID)

if training_metadata.get("status") != "completed":
    raise ValueError(f"Recommender training for {RUN_ID!r} is not marked completed: {training_metadata.get('status')!r}")

if evaluation_summary.get("status") not in {None, "completed", "evaluated"}:
    raise ValueError(f"Unexpected evaluation status for {RUN_ID!r}: {evaluation_summary.get('status')!r}")

if training_metadata.get("eval_instance_count", 0) <= 0:
    raise ValueError(f"Run {RUN_ID!r} has no held-out evaluation instances recorded in training_metadata.json")

TRAIN_DIR
"""
)

code(
    """display(build_title_block(run_info, training_metadata, SELECTION_ID, PERSONA))

aggregate = dict(evaluation_summary.get("aggregate", {}))
aggregate.pop("dataset_index", None)

run_overview = pd.DataFrame(
    {
        "run_id": [RUN_ID],
        "selection_id": [SELECTION_ID],
        "persona": [PERSONA],
        "model_key": [RECOMMENDER_MODEL_KEY],
        "alpha": [run_info.get("alpha")],
        "num_clients": [run_info.get("num_clients")],
        "top_k": [training_metadata.get("config", {}).get("top_k", [])],
        "rounds_completed": [training_metadata.get("rounds_completed")],
        "simulation_backend": [training_metadata.get("simulation_backend_actual")],
        "train_instances": [training_metadata.get("instance_count")],
        "test_instances": [training_metadata.get("eval_instance_count")],
        "train_candidates": [training_metadata.get("candidate_count")],
        "test_candidates": [training_metadata.get("eval_candidate_count")],
        "train_pairs_raw": [training_metadata.get("raw_pair_count")],
        "test_pairs_raw": [training_metadata.get("eval_raw_pair_count")],
        "train_pairs_augmented": [training_metadata.get("pair_count")],
        "test_pairs_augmented": [training_metadata.get("eval_pair_count")],
        "aggregate_pearson": [aggregate.get("pearson")],
        "aggregate_precision_at_1": [aggregate.get("precision_at_1")],
        "aggregate_precision_at_3": [aggregate.get("precision_at_3")],
        "aggregate_precision_at_5": [aggregate.get("precision_at_5")],
    }
)

run_overview
"""
)

code(
    """client_eval_df = pd.DataFrame(evaluation_summary.get("clients", []))
if client_eval_df.empty:
    raise ValueError(f"Run {RUN_ID!r} has no per-client evaluation results in evaluation_summary.json")

client_train_df = pd.DataFrame(training_metadata.get("clients", []))
if not client_train_df.empty:
    client_train_df = client_train_df.rename(
        columns={
            "candidate_count": "train_candidate_count",
            "instance_count": "train_instance_count",
            "raw_pair_count": "train_pair_count_raw",
            "augmented_pair_count": "train_pair_count_augmented",
        }
    )
    keep_columns = [
        "client_id",
        "train_candidate_count",
        "train_instance_count",
        "train_pair_count_raw",
        "train_pair_count_augmented",
    ]
    client_df = client_eval_df.merge(client_train_df[keep_columns], on="client_id", how="left")
else:
    client_df = client_eval_df.copy()

metric_columns = safe_metric_columns(client_df)
client_df["average_metric_score"] = client_df[metric_columns].mean(axis=1, skipna=True)
client_df["test_pairs_per_instance"] = client_df["pair_count"] / client_df["instance_count"]
client_df["test_candidates_per_instance"] = client_df["candidate_count"] / client_df["instance_count"]

client_df[
    [
        "client_id",
        "instance_count",
        "candidate_count",
        "pair_count",
        *metric_columns,
        "average_metric_score",
        "train_instance_count",
        "train_candidate_count",
        "train_pair_count_raw",
    ]
].sort_values("average_metric_score", ascending=False)
"""
)

code(
    """aggregate_metric_frame = pd.DataFrame([summarize_metric(client_df, metric) for metric in metric_columns])
aggregate_metric_frame
"""
)

code(
    """top_clients = client_df.sort_values("average_metric_score", ascending=False).head(3)
bottom_clients = client_df.sort_values("average_metric_score", ascending=True).head(3)

print("Top clients by average held-out score")
display(top_clients[["client_id", *metric_columns, "average_metric_score", "candidate_count", "pair_count"]].round(4))

print("Lowest clients by average held-out score")
display(bottom_clients[["client_id", *metric_columns, "average_metric_score", "candidate_count", "pair_count"]].round(4))
"""
)

code(
    """summary_columns = [
    "client_id",
    "instance_count",
    "candidate_count",
    "pair_count",
    *metric_columns,
    "average_metric_score",
    "test_candidates_per_instance",
    "test_pairs_per_instance",
    "train_instance_count",
    "train_candidate_count",
    "train_pair_count_raw",
]

client_report = client_df[summary_columns].sort_values("average_metric_score", ascending=False)
client_report.round(4)
"""
)

code(
    """fig, axes = plt.subplots(1, len(metric_columns), figsize=(4.5 * len(metric_columns), 4), sharey=False)
if len(metric_columns) == 1:
    axes = [axes]

ordered = client_df.sort_values("average_metric_score", ascending=False)
for ax, metric in zip(axes, metric_columns):
    ax.bar(ordered["client_id"], ordered[metric], color="#2E86AB", alpha=0.9)
    ax.axhline(ordered[metric].mean(), color="#C73E1D", linestyle="--", linewidth=2, label="mean")
    ax.set_title(metric)
    ax.set_xlabel("client")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

plt.suptitle("Held-out test metrics by client", y=1.05, fontsize=14)
plt.tight_layout()
plt.show()
"""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

aggregate_metric_frame.set_index("metric")[["mean", "std"]].plot(
    kind="bar",
    ax=axes[0],
    color=["#2E86AB", "#F18F01"],
    title="Mean and standard deviation across clients",
)
axes[0].tick_params(axis="x", rotation=45)
axes[0].set_ylabel("score")

aggregate_metric_frame.set_index("metric")[["min", "max"]].plot(
    kind="bar",
    ax=axes[1],
    color=["#7B2CBF", "#2A9D8F"],
    title="Smallest and highest client score",
)
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_ylabel("score")

plt.tight_layout()
plt.show()
"""
)

md(
    """## Notes

- This notebook reads the **final post-training** recommender evaluation from `evaluation_summary.json`.
- The reported metrics are for the **held-out test split** only.
- If a `run_id` does not have completed recommender artifacts, the loading cell raises a clear error with the missing files.
"""
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "fed-perso-xai",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("notebooks/recommender_evaluation_eda.ipynb").write_text(
    json.dumps(notebook, indent=1) + "\n",
    encoding="utf-8",
)
