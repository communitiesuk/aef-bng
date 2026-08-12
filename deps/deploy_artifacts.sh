#!/bin/bash
set -e

# --- Load Config ---
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
else
  echo "Error: .env file not found. Copy .env.example and fill in your values."
  exit 1
fi

if [ -z "$DATABRICKS_HOST" ]; then
    echo "Error: DATABRICKS_HOST must be set in .env file."
    exit 1
fi

# --- Paths ---
PROJECT_ROOT=$(pwd)
DIST_DIR="$PROJECT_ROOT/dist"
INIT_SCRIPT_DIR="$PROJECT_ROOT/deps/init-scripts"
WHEEL_FILE=$(find "$DIST_DIR" -name "*.whl" | head -n 1)

# --- Authenticate & Upload ---
echo "Deploying to: $DATABRICKS_HOST"

echo "Uploading init scripts to workspace..."
databricks workspace import-dir "$INIT_SCRIPT_DIR" "$DATABRICKS_INIT_SCRIPT_PATH" --overwrite

echo "Uploading Python wheel to to workspace..."
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel found in $DIST_DIR. Run 'make build' first."
    exit 1
fi
databricks workspace import-dir "$DIST_DIR" "$DATABRICKS_INIT_SCRIPT_PATH" --overwrite

echo "Uploading Python wheel to Unity Catalog..."
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel found in $DIST_DIR. Run 'make build' first."
    exit 1
fi
databricks fs cp "$WHEEL_FILE" "$DATABRICKS_WHEEL_VOLUME_PATH/" --overwrite

echo "All artifacts deployed."
