# fed-perso-xai

`fed-perso-xai` is the stage-1 baseline for federated Perso-XAI. This repository currently implements only the predictive-model training stage:

- OpenML-backed tabular dataset loading for Adult Income and Bank Marketing
- frozen global preprocessing for consistent feature dimensionality across clients
- Dirichlet client partitioning with per-client train/test splits saved as `.npz`
- Flower-based federated logistic regression with weighted evaluation reporting

The later Perso-XAI stages are not implemented here yet. The code is organized so explanation generation, explanation metrics, and recommender/clustering modules can be added as independent pipeline stages without rewriting the data or training baseline.

**Structure**

- `src/fed_perso_xai/data`: dataset loading, preprocessing, partitioning, serialization
- `src/fed_perso_xai/models`: explicit NumPy logistic regression
- `src/fed_perso_xai/fl`: Flower client, strategy factory, sequential simulation fallback
- `src/fed_perso_xai/evaluation`: per-client metrics and weighted aggregation
- `src/fed_perso_xai/orchestration`: stage-oriented pipeline runners
- `tests`: smoke and unit coverage for stage-1 behavior

**Install**

```bash
python3 -m pip install -e .
```

**Prepare Data**

This command:

- downloads or loads the selected OpenML dataset
- creates a global raw-data train pool and held-out global evaluation pool
- fits a single preprocessing schema on the global raw-data train pool
- transforms the global train pool once and partitions it across clients
- creates each client's local train/test split
- writes client arrays and metadata to disk

```bash
python3 -m fed_perso_xai prepare-data \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --seed 42
```

Saved client partitions follow the required layout under the configured output root:

```text
datasets/10_clients/alpha_1/client_0/train.npz
datasets/10_clients/alpha_1/client_0/test.npz
...
datasets/10_clients/alpha_1/client_9/train.npz
datasets/10_clients/alpha_1/client_9/test.npz
```

Additional artifacts written under the partition root:

- `metadata.json`: dataset name, preprocessing schema summary, feature names, class distributions
- `global_eval.npz`: optional centralized holdout transformed with the same frozen schema

**Train Federated Baseline**

```bash
python3 -m fed_perso_xai train-federated \
  --dataset adult_income \
  --num-clients 10 \
  --alpha 1.0 \
  --rounds 20 \
  --local-epochs 5 \
  --batch-size 64 \
  --learning-rate 0.05 \
  --seed 42
```

Training writes results under `results/<dataset>/<K>_clients/alpha_<alpha>/seed_<seed>/`:

- `metrics_summary.json`
- `model_parameters.npz`

The metrics summary includes:

- per-client loss, accuracy, precision, recall, F1, and ROC-AUC when defined
- weighted aggregated metrics across client test splits
- round-level history from the stage-1 federated baseline

**Design Notes**

- Logistic regression is implemented explicitly in NumPy so Flower parameter exchange is simple and inspectable.
- The current environment does not include Ray, so the baseline uses a Flower-native sequential fallback built around `FedAvg`. The strategy seam is isolated so a future run can switch to native Flower simulation or a custom secure/clustered strategy.
- Secure aggregation and clustering are intentionally not implemented yet. The `StrategyFactory` abstraction in `fl/strategy.py` is the extension point for later secure or clustered reducers inspired by `lcc-lib`.
- Explanation generation, metric computation beyond predictive evaluation, and personalized recommendation are intentionally separate future pipeline stages.

**Known Limitations**

- Only binary tabular classification is supported in stage 1.
- Only Adult Income and Bank Marketing are wired in.
- Native Flower simulation with Ray is not enabled unless `ray` is installed alongside Flower.
- OpenML downloads require network access on first use unless the cache is already populated.
