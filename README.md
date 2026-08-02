# Football match prediction

Kicktipp-oriented Bundesliga prediction with CatBoost Poisson models, season holdout evaluation, and local MLflow tracking.

## Setup

```zsh
uv sync
```

Create a `.env` for Kicktipp upload / scheduler email (see below).

## Season holdout

Default: train on seasons **before** 2025, evaluate on **2025** (2025/26).

```zsh
uv run python scripts/create_datasets.py --holdout-season 2025
uv run python scripts/train.py --holdout-season 2025
uv run python -m eval.compare --holdout-season 2025 --baseline majority
```

Form/Elo are built causally over train+holdout history; model weights fit on train labels only.

## MLflow

```zsh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Training logs:
- **Datasets** (train + holdout) via `mlflow.log_input`
- **pyfunc model** + **Model Registry** entry `bundesliga-kicktipp` with alias `@candidate`

```zsh
# Promote candidate into local artifacts/ and mark @production
uv run python scripts/promote_run.py --alias candidate --backup --set-production-alias

# Or by URI / run id
uv run python scripts/promote_run.py --model-uri 'models:/bundesliga-kicktipp@candidate' --backup
uv run python scripts/promote_run.py --run-id <RUN_ID> --backup
```

## Predict / upload

```zsh
uv run python scripts/predict.py
uv run python scripts/upload_predictions.py          # fill only
uv run python scripts/upload_predictions.py --submit # fill + submit
```

## Scheduler

```zsh
uv run python run_scheduler.py
# or
docker compose up -d
```

## Reload historical match data

If `data/match_df_*.pck` is missing:

```zsh
uv run python scripts/reload_seasons.py --from-year 2009 --to-year 2025 --leagues bl1 bl2
```

Market values fall back to `datasets/TeamMarketValues.csv` if `data/market_values_dict.pck` is missing.

## Layout

```
api/                 # OpenLigaDB / Transfermarkt clients
artifacts/           # Production model bundle
data/                # match_df_*.pck, market values
datasets/            # train.csv / holdout test.csv
eval/                # metrics, baselines, compare, mlflow helpers
scripts/             # train, predict, scheduler, …
model.py
evaluation.py        # thin wrapper → eval.compare
```

## Environment

```env
EMAIL=...
PASSWORT=...
LINK-TIPPABGABE=https://www.kicktipp.de/.../tippabgabe
EMAIL_RECEIVER=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
```
