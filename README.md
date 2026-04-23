# fed-perso-xai

`fed-perso-xai` is the baseline implementation for the larger federated Perso-XAI project. This repository is intentionally limited to predictive-model training with logistic regression, but its data, evaluation, and artifact contracts are shaped for later explanation-generation and recommender work.

Implemented now:

- dataset preparation for supported tabular datasets
- frozen preprocessing fitted once on the global raw training pool
- centralized logistic-regression training
- Flower-based federated simulation as the primary federated path
- explicit prediction, metrics, manifest, and provenance artifacts
- centralized-versus-federated comparison reporting

Not implemented yet:

- explanation generation
- explanation metrics
- recommender training
- pairwise comparison or preference learning
- clustering

## Repository Layout

- `src/fed_perso_xai/data`: dataset specs, loaders, preprocessing, partitioning, serialization
- `src/fed_perso_xai/models`: logistic-regression baseline
- `src/fed_perso_xai/fl`: Flower client, strategy, and simulation runtime integration
- `src/fed_perso_xai/evaluation`: predictive metrics, prediction artifacts, comparison reports
- `src/fed_perso_xai/orchestration`: prepare-data and training entrypoints
- `tests`: smoke and contract tests for the baseline

## Installation

Base installation for data preparation, centralized training, and artifact inspection:

```bash
python3 -m pip install -e .
```

Optional dependency groups:

- `.[dev]`: test and local development tooling
- `.[fl]`: Flower support for federated code paths, including the explicit debug sequential runtime
- `.[ray]`: Ray-backed Flower simulation support for the primary federated runtime

Secure aggregation also needs the sibling `lcc-lib` package installed:

```bash
python3 -m pip install -e ../lcc-lib
```

Typical setups:

```bash
python3 -m pip install -e ".[dev,fl]"
python3 -m pip install -e ".[dev,ray]"
```

If you only need data preparation and centralized training, the base install is sufficient:

```bash
python3 -m pip install -e .
```

`main.py` remains available for convenience:

```bash
python3 main.py --help
```

## Supported Datasets

- `adult_income`
- `bank_marketing`

Both are loaded from OpenML, normalized to binary labels `{0,1}`, and passed through the shared tabular preprocessing pipeline.

## Adding A New Dataset

The intended extension path is to define a new `DatasetSpec`, optionally register a dataset-specific cleaning hook, and then reuse the existing orchestration.

A new dataset usually needs:

1. a `DatasetSpec` with `key`, `display_name`, `openml_data_id`, `target_column`, and `target_transform`
2. optional `cleaning_hook` for dataset-specific cleanup before generic preprocessing
3. optional `feature_type_overrides`
4. optional `required_columns`

The preparation, centralized training, federated training, and comparison flows are meant to continue working without editing their core logic.

## Stage-1 Workflow

### 1. Prepare Data

```bash
python3 -m fed_perso_xai prepare-data \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --seed 42
```

This step:

- loads raw data
- creates the global raw train/eval split
- fits the frozen preprocessor on the raw global training pool only
- saves transformed global train/eval arrays
- partitions the transformed training pool into clients
- creates each client train/test split
- writes split and partition metadata for downstream runs

Stage-1 preprocessing assumption:

- `fitting_mode=global_shared` is explicit in config and metadata
- the frozen preprocessor is fit once on the global raw training pool before partitioning
- this is an intentional baseline assumption for comparability, not a privacy-faithful local preprocessing flow

Prepared artifacts are written under:

```text
prepared/<dataset>/seed_<seed>/
```

Key prepared artifacts:

- `prepare_config.json`
- `dataset_metadata.json`
- `split_metadata.json`
- `feature_metadata.json`
- `preprocessor.joblib`
- `global_train.npz`
- `global_eval.npz`
- `pooled_client_test.npz`
- `partition_metadata.json`

Client partitions are written under:

```text
datasets/<dataset>/<K>_clients/alpha_<alpha>/seed_<seed>/
```

with:

```text
client_<id>/train.npz
client_<id>/test.npz
partition_metadata.json
```

### 2. Train Centralized Baseline

```bash
python3 -m fed_perso_xai train-centralized \
  --dataset adult_income \
  --seed 42 \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.05
```

Outputs are written under:

```text
centralized/<dataset>/seed_<seed>/
```

Important artifacts:

- `run_manifest.json`
- `config_snapshot.json`
- `reproducibility_metadata.json`
- `metrics_summary.json`
- `model_parameters.npz`
- `predictions_global_eval.npz`
- `predictions_pooled_client_test.npz` when enabled
- copied shared artifacts:
  - `preprocessor.joblib`
  - `feature_metadata.json`
  - `dataset_metadata.json`
  - `split_metadata.json`
  - `partition_metadata.json`

`run_manifest.json` is the top-level pointer for the run and records the run id, mode, timestamp, important configuration, reproducibility fields, and the relative paths to the persisted artifacts in that run directory.

### 3. Train Federated Baseline

Primary runtime, using real Flower simulation with Ray:

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --rounds 20 \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.05 \
  --seed 42 \
  --simulation-backend ray
```

Auto mode resolves to the primary Flower path when Ray is installed:

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --seed 42 \
  --simulation-backend auto
```

Debug-only sequential runtime:

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --seed 42 \
  --simulation-backend debug-sequential
```

Runtime policy:

- real Flower simulation is the intended federated path
- `auto` does not silently downgrade to sequential execution when Ray is missing
- `debug-sequential` is explicit and reported as a development fallback
- `--debug-fallback-on-error` enables an explicit fallback if a Ray simulation fails and you still want the debug runtime

Federated outputs are written under:

```text
federated/<dataset>/<K>_clients/alpha_<alpha>/seed_<seed>/
```

Important artifacts:

- `run_manifest.json`
- `config_snapshot.json`
- `reproducibility_metadata.json`
- `runtime_report.json`
- `metrics_summary.json`
- `model_parameters.npz`
- `predictions_client_test.npz`
- copied shared artifacts:
  - `preprocessor.joblib`
  - `feature_metadata.json`
  - `dataset_metadata.json`
  - `split_metadata.json`
  - `partition_metadata.json`

### Aggregation Modes

The federated strategy now supports two server-side aggregation modes over the
same shared/global tensors:

- plain weighted averaging
- protocol-first secure aggregation using in-process simulated helpers

The baseline model has no personalized server-excluded tensors yet, so the
client adapter currently marks all model parameters as shared. The extraction
and merge helpers are still explicit so later phases can keep local/personal
parameters on-device while only aggregating the shared subset.

Secure aggregation flow in this prototype:

1. clients train locally and return the shared parameter payload
2. the strategy extracts the shared tensors from each client result
3. `lcc-lib` flattens, quantizes, secret-shares, and uploads one share per helper
4. in-memory helper objects sum shares for the round
5. the server reconstructs the aggregate, dequantizes it, restores tensor shapes, and applies the usual weighted average semantics

No gRPC, HTTP, or external helper process is involved in this phase.

Plain aggregation example:

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --rounds 10 \
  --seed 42 \
  --simulation-backend debug-sequential
```

Secure aggregation example:

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --rounds 10 \
  --seed 42 \
  --simulation-backend debug-sequential \
  --secure-aggregation \
  --secure-num-helpers 5 \
  --secure-privacy-threshold 2 \
  --secure-reconstruction-threshold 3 \
  --secure-field-modulus 2147483647 \
  --secure-quantization-scale 65536 \
  --secure-seed 7
```

Equivalent config snapshot fields inside a federated run:

```json
{
  "secure_aggregation": false
}
```

```json
{
  "secure_aggregation": true,
  "secure_num_helpers": 5,
  "secure_privacy_threshold": 2,
  "secure_reconstruction_threshold": 3,
  "secure_field_modulus": 2147483647,
  "secure_quantization_scale": 65536,
  "secure_seed": 7
}
```

## Artifact Contract

Prediction artifacts include at least:

- `run_id`
- `dataset_name`
- `split_name`
- `y_true`
- `y_pred`
- `y_prob`
- `row_ids`
- `client_ids` when applicable

Feature metadata includes:

- raw columns expected, kept, and dropped
- dropped-column reasons
- grouped diagnostics for dropped constant and all-missing columns
- numeric and categorical column groups
- imputed columns
- stable transformed feature ordering
- transformed feature names
- transformed-to-raw and raw-to-transformed maps
- feature lineage records
- encoder vocabularies
- schema diagnostics
- unknown-category handling policy

Each training run emits a `run_manifest.json` tying together the major artifacts so later explanation-generation, evaluation, and recommender stages can consume them without redefining the contract.

## Comparison Reports

`compare-baselines` compares predictive modeling outputs only:

```bash
python3 -m fed_perso_xai compare-baselines \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --seed 42
```

The report includes:

- centralized global-eval metrics
- centralized pooled-client-test metrics when available
- federated weighted client-test metrics
- federated pooled client-test metrics
- absolute metric differences for shared metrics
- metric availability notes when a metric is missing from one split
- source run directories and manifests
- federated per-client metric summaries
- split provenance
- class-balance summaries
- probability summaries
- metric deltas across modes

The report structure is intended to accept future explanation or recommender metrics as additional sections instead of requiring a redesign.

## Tests

Run the baseline test suite with:

```bash
python3 -m pytest -q
```

Current tests cover:

- dataset registry extensibility with a synthetic dataset spec
- preprocessing robustness for constant, all-missing, unseen-category, and boolean/categorical cases
- prepared artifact generation
- partitioning reproducibility
- centralized and federated artifact compatibility
- comparison report generation
- real Flower simulation smoke execution when the `ray` extra is installed
