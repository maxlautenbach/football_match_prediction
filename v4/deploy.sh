#!/bin/bash
# Script to create a tar archive of v4 and deploy to CapRover

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create tar archive excluding unnecessary files
echo "Creating tar archive..."
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

echo "Tar archive created: ../v4.tar.gz"
echo ""

# Remove any existing -t flags from arguments and add our own
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

# Deploy to CapRover with all provided flags (except -t)
echo "Deploying to CapRover..."
caprover deploy -t ../v4.tar.gz "${DEPLOY_ARGS[@]}"

echo ""
echo "Deployment completed!"

