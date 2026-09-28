# Football match prediction

Kicktipp-oriented Bundesliga prediction with CatBoost Poisson and Dixon-Coles models, season holdout evaluation, and local MLflow tracking.

## Setup

```zsh
uv sync
```

Create a `.env` for Kicktipp upload / scheduler email (see below).

## Season holdout

Default: train on seasons **before** 2025, evaluate on **2025** (2025/26).

```zsh
uv run python scripts/create_datasets.py --holdout-season 2025

# Train into a temp bundle → MLflow (does NOT overwrite artifacts/)
uv run python scripts/train.py --recipe recipes/catboost_poisson.toml --holdout-season 2025
uv run python scripts/train.py --recipe recipes/dixon_coles.toml --holdout-season 2025
uv run python scripts/train.py --recipe recipes/majority_baseline.toml --holdout-season 2025

uv run python -m eval.compare --holdout-season 2025 --baseline majority
```

See [`docs/MODELS.md`](docs/MODELS.md) and [`AGENTS.md`](AGENTS.md) for the models/ + recipes/ convention.

Form/Elo are built causally over train+holdout history; model weights fit on train labels only.

## Saison outlook (separate experiment)

Season-start Kicktipp tips (Meister / Herbstmeister / Plätze 16–18 / Torjäger-Mannschaft), 6 points each (max 24). Holdout season 2025. Score-maxing MV/history priors — not the match `"H:A"` model.

```zsh
uv run python scripts/reload_goals.py --start 2009 --end 2025
uv run python scripts/create_saison_dataset.py --holdout-season 2025
uv run python scripts/train_saison.py \
  --recipe recipes/saison_ausblick.toml \
  --holdout-season 2025
```

Logs to MLflow experiment `kicktipp-saison`, registry `kicktipp-saison-ausblick` `@candidate`. Does not touch match `artifacts/`.

## MLflow

```zsh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Artifact files live under `mlruns/` (gitignored) — do not delete that folder while using the local registry.

Training logs:
- **Datasets** (train + holdout) via `mlflow.log_input`
- **Poisson blend pyfunc** → `kicktipp-poisson-blend` `@candidate` / `@production` (current production)
- **CatBoost pyfunc** → `kicktipp-catboost-poisson` `@candidate` / `@production`
- **Dixon-Coles pyfunc** → `kicktipp-dixon-coles` `@candidate`
- **Market Dixon-Coles** → `kicktipp-market-dixon-coles` `@candidate`
- **Majority baseline** → `kicktipp-majority-baseline` `@baseline`
- **Saison outlook** → experiment `kicktipp-saison`, `kicktipp-saison-ausblick` `@candidate`

Match experiment: `kicktipp` (`sqlite:///mlflow.db`).

```zsh
# Promote poisson_blend candidate into local artifacts/ and mark @production
uv run python scripts/promote_run.py \
  --alias candidate \
  --registered-model kicktipp-poisson-blend \
  --backup \
  --set-production-alias

# Or by URI / run id
uv run python scripts/promote_run.py --model-uri 'models:/kicktipp-poisson-blend@candidate' --backup
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

Goal events for saison outlook (`data/goals_df_*.pck`):

```zsh
uv run python scripts/reload_goals.py --start 2009 --end 2025
```

Market values fall back to `datasets/TeamMarketValues.csv` if `data/market_values_dict.pck` is missing.

## Layout

```
api/                 # OpenLigaDB / Transfermarkt clients
artifacts/           # Production model bundle (promote only)
data/                # match_df_*.pck, goals_df_*.pck, market values
datasets/            # train.csv / holdout test.csv / saison_*.csv
docs/MODELS.md       # model convention (authoritative)
models/              # model types (catboost_poisson, dixon_coles, saison_ausblick, …)
recipes/             # TOML recipes for training
eval/                # metrics, baselines, compare, mlflow helpers
scripts/             # train, train_saison, promote, predict, scheduler, …
model.py             # thin public Model adapter
evaluation.py        # thin wrapper → eval.compare
```

## Environment

Copy `.env.example` → `.env` locally. On CapRover, set the same keys as app env vars (do not bake `.env` into the image).

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

After each tip job the scheduler sends an HTML digest (tips for the next matchday plus
performance for the previous archived matchday: Kicktipp points, expected points, Δ, z-score).
Tips are stored under `data/tips/` — keep that directory on the persistent `data` volume.

## CapRover deploy

Production image runs `run_scheduler.py` (predict + Kicktipp upload on schedule). Image includes:

- `artifacts/` — current production bundle (`poisson_blend`)
- `data/` — match pickles (incl. season 2026)
- `models/` — required to load the bundle

```zsh
# 1) CapRover app env: EMAIL, PASSWORT, LINK-TIPPABGABE (+ optional SMTP_*)
# 2) Optional persistent dirs: /app/data and /app/artifacts
#    (entrypoint seeds them from the image if empty)

./deploy.sh
# or: ./deploy.sh -a your-app-name
```

`deploy.sh` builds a tarball (excludes `.env`, MLflow, backups) and runs `caprover deploy`.

On boot the scheduler refreshes the current season delta, runs one tip+upload job
(unless `SCHEDULER_SKIP_STARTUP_JOB=1`), then waits for the next matchday (+3h after
the last kickoff).
