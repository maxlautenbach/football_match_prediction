# Football match prediction (UV)

## Overview

v4 is a standalone prediction system with Delta-Logic (Option A) for efficient data updates.
All scripts are located in `scripts/` and data in `data/`.

## Delta-Logic (Option A)

The system uses incremental updates (Delta-Logic Option A):

- Only new or changed matches are loaded from the API
- Individual API calls per outdated match (minimal traffic)
- Historical seasons are never updated (final data)
- Current season is checked for outdated "future" matches

## Setup

```zsh
cd v4
uv sync
```

## Scripts

All scripts are located in `scripts/`:

### 1. Create Datasets

Generate CSV datasets from pickle files:

```zsh
uv run python scripts/create_datasets.py
```

This script:

- Runs delta update (Option A) to get latest matches
- Loads all `match_df_*.pck` files from `data/`
- Generates `train.csv`, `test.csv`, `TeamMarketValues.csv` in `datasets/`

### 2. Train

Train the model:

```zsh
uv run python scripts/train.py
```

This script:

- Runs delta update (Option A) before training
- Generates datasets dynamically from pickle files
- Trains CatBoost models for home/away goals
- Saves artifacts to `artifacts/`

### 3. Predict

Make predictions for next matchday:

```zsh
uv run python scripts/predict.py
```

This script:

- Runs delta update (Option A) to get latest next_matchday
- Loads model from `artifacts/`
- Makes predictions and saves to CSV (e.g., `prediction_YYYYMMDD_HHMMSS.csv`)

### 4. Upload Predictions

Upload predictions to betting platform:

```zsh
uv run python scripts/upload_predictions.py
```

Options:

- `--submit`: Automatically submit predictions after filling (default: manual review)

This script:

- Generates predictions directly from `predict.py` (no CSV files needed)
- Uploads predictions to betting platform (requires `.env` file)
- Optionally auto-submits or waits for manual review

### 5. Evaluate

Evaluate model on test set:

```zsh
uv run python evaluation.py
```

`evaluation.py` loads `datasets/test.csv`, calls `Model.predict(...)`, prints metrics, and writes a `prediction_*.csv`.

### 6. Scheduler (Server Mode)

Run the automated scheduler service:

```zsh
uv run python run_scheduler.py
```

This starts the scheduler service that:

- Runs a test job immediately on startup
- Automatically detects when a matchday ends
- Schedules the next prediction job (matchday_end + 1 day, 17:00)
- Sends email notifications with logs after each job
- Runs continuously, checking for new matchdays every 6 hours

## Docker Deployment

The application can be deployed as a Docker container with automatic scheduling.

### Prerequisites

1. Docker and Docker Compose installed
2. `.env` file configured (see [Environment Variables](#environment-variables))
3. `data/` and `artifacts/` directories with required files

### Build and Run

```zsh
cd v4
docker-compose up -d
```

This will:

- Build the Docker image with all dependencies
- Start the scheduler service
- Run a test job immediately
- Schedule future prediction jobs automatically

### View Logs

```zsh
docker-compose logs -f
```

### Stop Service

```zsh
docker-compose down
```

### Manual Build (without docker-compose)

```zsh
# Build from parent directory (context needed for api/ folder)
cd ..
docker build -f v4/Dockerfile -t football-prediction:v4 .

# Run container
docker run -d \
  --name football-prediction \
  -v $(pwd)/v4/data:/app/data \
  -v $(pwd)/v4/artifacts:/app/artifacts \
  --env-file v4/.env \
  football-prediction:v4
```

## CapRover Deployment

To deploy to CapRover, you need to create a tar archive since CapRover's `deploy` command requires either a git repository or a tar file when not in a git root.

### Quick Deploy

Use the provided deployment script:

```zsh
cd v4
./deploy.sh
caprover deploy -t ../v4.tar.gz -a YOUR_APP_NAME
```

### Manual Deploy

1. **Create tar archive:**

```zsh
cd v4
tar -czf ../v4.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='node_modules' \
    --exclude='.git' \
    --exclude='*.log' \
    --exclude='prediction_*.csv' \
    --exclude='.DS_Store' \
    .
```

2. **Deploy to CapRover:**

```zsh
caprover deploy -t ../v4.tar.gz -a YOUR_APP_NAME
```

Or if you're already configured:

```zsh
caprover deploy -t ../v4.tar.gz
```

### CapRover Configuration

The `captain-definition` file is configured to use `Dockerfile.caprover`, which is optimized for CapRover's build context (v4 directory only). The `api/` directory is included in v4 for CapRover deployment.

**Note:** Make sure your CapRover app has persistent volumes configured for `/app/data` and `/app/artifacts` if you want data to persist across deployments.

### How the Scheduler Works

1. **On Container Start:**

   - Runs a test prediction job immediately
   - Detects the last completed matchday
   - Schedules the next job (matchday_end + 1 day, 17:00)
   - Sends email with logs and next job time

2. **After Each Matchday:**

   - Detects when all matches of a matchday are finished
   - Schedules next prediction job automatically
   - Job runs: `predict.py` → `upload_predictions.py --submit`
   - Sends email notification with full logs

3. **Continuous Monitoring:**
   - Checks for new completed matchdays every 6 hours
   - Automatically schedules new jobs as needed

## Data Structure

```
v4/
├── scripts/           # All executable scripts
│   ├── train.py      # Training script
│   ├── predict.py    # Prediction script
│   ├── upload_predictions.py  # Upload to betting platform
│   ├── create_datasets.py  # Dataset generation
│   ├── data_loader.py # Delta-Logic (Option A)
│   └── scheduler.py  # Scheduler service (APScheduler)
├── data/             # Pickle files (from v3/data)
│   ├── match_df_*.pck
│   ├── market_values_dict.pck
│   └── next_matchday_df.pck
├── datasets/         # Generated CSV files
├── artifacts/        # Model artifacts
├── model.py          # Model class
├── evaluation.py     # Evaluation script
├── run_scheduler.py  # Scheduler entry point
├── Dockerfile        # Docker container definition
├── docker-compose.yml # Docker Compose configuration
└── .dockerignore     # Docker build exclusions
```

## Environment Variables

Create a `.env` file in the `v4/` directory with the following variables:

### Required for Prediction Upload

These variables are needed if you want to automatically upload predictions to the betting platform:

```
EMAIL=your_email@example.com
PASSWORT=your_password
LINK-TIPPABGABE=https://www.kicktipp.de/your-league/tippabgabe
```

- `EMAIL`: Your login email for the betting platform
- `PASSWORT`: Your login password for the betting platform
- `LINK-TIPPABGABE`: Full URL to the prediction submission page

### Required for E-Mail Logging

These variables are needed for the scheduler to send email notifications:

```
EMAIL_RECEIVER=recipient@example.com
```

- `EMAIL_RECEIVER`: Email address where job logs will be sent

### Optional SMTP Configuration

If you need to use a custom SMTP server (defaults to Gmail):

```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_smtp_user@example.com
SMTP_PASSWORD=your_smtp_app_password
```

- `SMTP_SERVER`: SMTP server hostname (default: `smtp.gmail.com`)
- `SMTP_PORT`: SMTP server port (default: `587`)
- `SMTP_USER`: SMTP username for authentication (optional, uses `EMAIL_RECEIVER` if not set)
- `SMTP_PASSWORD`: SMTP password or app password (required if `SMTP_USER` is set)

### Example `.env` File

```env
# Betting Platform Credentials
EMAIL=your_email@example.com
PASSWORT=your_password
LINK-TIPPABGABE=https://www.kicktipp.de/your-league/tippabgabe

# E-Mail Logging
EMAIL_RECEIVER=notifications@example.com

# Optional: Custom SMTP (for Gmail, use app password)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

**Note for Gmail users:** You need to generate an [App Password](https://support.google.com/accounts/answer/185833) and use it as `SMTP_PASSWORD` instead of your regular password.

## Delta-Logic Details

**Option A Implementation:**

- For each outdated match: `openligadb.get_match_data(match_id)`
- Only changed matches are loaded
- Minimal API traffic
- Error handling per match (one failure doesn't block others)

**Current Season Detection:**

- August-December: Season started in current year
- January-July: Season started in previous year

**Update Process:**

1. Determine current season
2. Load existing `match_df_{current_season}.pck`
3. Find matches with `status="future"` that are in the past
4. For each outdated match: Load from API individually
5. Update pickle file with delta
6. Update `next_matchday_df.pck`
