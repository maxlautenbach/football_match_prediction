# Model development convention

This document is mandatory for humans and coding agents adding or changing
football prediction models.

## Public contract

Production callers import `Model` from the repository-root `model.py`.

- `Model()` loads the bundle in `artifacts/`.
- `Model(artifacts_dir=...)` loads a specified bundle.
- `Model.predict(X)` returns one `"H:A"` string per input row.
- Prediction order must match input order.
- Required input columns are `Team Home`, `Team Away`, `Saison`,
  `Spieltag`, and `Wochentag`.
- Loading failures and malformed predictions must fail explicitly. Production
  must not silently substitute `0:0`.

The root `model.py` is a stable adapter. Algorithm-specific code belongs in
`models/<model_type>/`.

## Layout

```text
models/
  contract.py                 # Predictor protocol + "H:A" validation + bundle.json helpers
  registry.py                 # model_type → train / load
  common/kicktipp.py          # shared Kicktipp Poisson / expected-points decode
  common/teams.py             # shared team name normalization
  catboost_poisson/           # main production model
  dixon_coles/                # time-decayed bivariate Poisson ratings
  majority_baseline/          # majority-class baseline
recipes/
  catboost_poisson.toml
  dixon_coles.toml
  majority_baseline.toml
scripts/train.py              # generic orchestrator
scripts/promote_run.py        # validate + atomic promote into artifacts/
artifacts/                    # production bundle only
model.py                      # thin public adapter
```

## Model types and recipes

Each model type has a stable lowercase snake_case identifier, for example
`catboost_poisson`.

A model type provides:

1. `train.py`: trains from supplied train/holdout data and writes a bundle.
2. `model.py`: loads that bundle and implements `predict(DataFrame)`.
3. Registration in `models/registry.py`.
4. At least one recipe in `recipes/<recipe_name>.toml`.

Recipes contain model parameters, not Python logic. Recipe names are stable
and descriptive.

Currently implemented:

- `catboost_poisson` — CatBoost home/away goal regressors (Poisson loss) over
  market values, team aggregates, form and Elo. Production model.
- `dixon_coles` — bivariate Poisson attack/defence ratings fitted by
  time-decayed maximum likelihood, with the Dixon-Coles low-score correction.
  Ratings are refitted at every matchday checkpoint from matches played strictly
  earlier, and stored as a lookup table in the bundle. Fitting spans BL1 **and**
  BL2 so promoted teams arrive with a real rating; teams without enough recent
  evidence fall back to a weak-team prior.
- `majority_baseline` — majority result of the training data.

`catboost_poisson` and `dixon_coles` both decode goal expectations into the
score maximizing expected Kicktipp points (`models/common/kicktipp.py`).

## Bundle contract

Every bundle contains `bundle.json`:

- `schema_version`
- `model_type`
- `recipe_name`
- `created_at`
- `holdout_season`
- `train_seasons`
- `required_columns`
- `files` (model-specific artifact file names)

CatBoost bundles also keep legacy `meta.json` (+ `.cbm` / `.joblib` files) for
backward compatibility during the transition.

`artifacts/` is the deployed bundle. Training must write to a temporary
working directory and log that directory to MLflow. Only
`scripts/promote_run.py` may replace `artifacts/`.

## Training

Canonical commands:

```bash
# Main model
uv run python scripts/train.py \
  --recipe recipes/catboost_poisson.toml \
  --holdout-season 2025 \
  --skip-delta

# Dixon-Coles ratings
uv run python scripts/train.py \
  --recipe recipes/dixon_coles.toml \
  --holdout-season 2025 \
  --skip-delta

# Baseline
uv run python scripts/train.py \
  --recipe recipes/majority_baseline.toml \
  --holdout-season 2025 \
  --skip-delta
```

The generic training script must:

1. Load and split data by season.
2. Resolve the recipe's `model_type` through `models/registry.py`.
3. Train into a temporary bundle directory.
4. Load the completed bundle through the public `Model` adapter.
5. Validate prediction count and `"H:A"` formatting.
6. Evaluate the fixed holdout with `eval.metrics`.
7. Log parameters, metrics, datasets, bundle, and pyfunc to MLflow.
8. Register the successful model version (`@candidate` or `@baseline`).
9. Leave `artifacts/` unchanged.

## MLflow convention

Use constants from `eval/mlflow_utils.py` (do not hardcode after renames):

- Experiment: `EXPERIMENT_NAME` (`kicktipp`)
- Main registered model: `REGISTERED_MODEL_NAME` (`kicktipp-catboost-poisson`)
- Dixon-Coles registered model: `DIXON_COLES_REGISTERED_MODEL_NAME` (`kicktipp-dixon-coles`)
- Baseline registered model: `BASELINE_REGISTERED_MODEL_NAME` (`kicktipp-majority-baseline`)
- Run name: `<model_type>-<recipe_name>-holdout-<season>`
- Model artifact path: `model`
- Candidate URI: `models:/kicktipp-catboost-poisson@candidate`
- Production URI: `models:/kicktipp-catboost-poisson@production`
- Baseline URI: `models:/kicktipp-majority-baseline@baseline`

Required run parameters/tags:

- `model_type`, `recipe_name`, `holdout_season`, `bundle_schema_version`
- training / holdout row counts
- relevant hyperparameters
- `git_commit` when available

Required metrics come from `eval.metrics`, including `kicktipp_score`,
`kicktipp_raw`, `exact_accuracy`, `goal_difference_accuracy`, and
`outcome_accuracy`.

Each model type has its own registered model name. Shared scoring contract
stays in root `model.py` + `models/contract.py`.

## Promotion

Compare the candidate against production on the same holdout, then run:

```bash
uv run python scripts/promote_run.py \
  --alias candidate \
  --backup \
  --set-production-alias
```

For the baseline registry:

```bash
uv run python scripts/promote_run.py \
  --alias baseline \
  --registered-model kicktipp-majority-baseline \
  --dst artifacts_baseline
```

(Only promote the main CatBoost model into production `artifacts/` used by
`scripts/predict.py` / the scheduler.)

Promotion must:

1. Download the selected registry version.
2. Validate `bundle.json` (or accept legacy `meta.json`).
3. Load it through `Model(artifacts_dir=...)`.
4. Run a smoke prediction.
5. Replace `artifacts/` atomically.
6. Set `@production` only after local promotion succeeds.

Do not manually copy files into `artifacts/` or set `@production` during
training.

## Adding a model type

1. Choose a unique `model_type`.
2. Add `models/<model_type>/train.py` and `model.py`.
3. Register it in `models/registry.py` (and add a default registered-name
   constant in `eval/mlflow_utils.py` if needed).
4. Add a TOML recipe under `recipes/`.
5. Train with the standard command.
6. Inspect the MLflow run and compare holdout metrics.
7. Promote only through `scripts/promote_run.py`.

Avoid duplicating data loading, holdout splitting, metrics, MLflow logging,
registry aliases, or promotion logic inside a model package.
