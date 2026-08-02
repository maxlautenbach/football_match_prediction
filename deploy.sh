#!/bin/bash
# Create a tar archive of the project and deploy to CapRover

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ARCHIVE="deploy.tar.gz"

echo "Creating tar archive..."
tar -czf "$ARCHIVE" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='node_modules' \
    --exclude='.git' \
    --exclude='*.log' \
    --exclude='prediction_*.csv' \
    --exclude='.DS_Store' \
    --exclude='mlflow.db' \
    --exclude='mlruns' \
    --exclude='mlartifacts' \
    --exclude='artifacts_prev' \
    --exclude="$ARCHIVE" \
    .

echo "Tar archive created: $ARCHIVE"
echo ""

DEPLOY_ARGS=()
SKIP_NEXT=false
for arg in "$@"; do
    if [ "$SKIP_NEXT" = true ]; then
        SKIP_NEXT=false
        continue
    fi
    if [ "$arg" = "-t" ] || [ "$arg" = "--tar" ]; then
        SKIP_NEXT=true
        continue
    fi
    DEPLOY_ARGS+=("$arg")
done

echo "Deploying to CapRover..."
caprover deploy -t "$ARCHIVE" "${DEPLOY_ARGS[@]}" --default

echo ""
echo "Deployment completed!"
