#!/bin/bash
set -e

# --- Install uv ---
echo "Downloading uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# --- Locate wheel ---
WHEEL_NAME="aef_bng-0.1.0-py3-none-any.whl" # Update with relevant wheel file name
SCRIPT_DIR=$(dirname "$0")
LOCAL_WHEEL_PATH="${SCRIPT_DIR}/${WHEEL_NAME}"
UC_WHEEL_PATH="/Volumes/catalog/schema/volume/${WHEEL_NAME}"

# --- Install wheel (try local directory first, then UC volume) ---
if [ -f "${LOCAL_WHEEL_PATH}" ]; then
  echo "Installing wheel from script directory: ${LOCAL_WHEEL_PATH}"
  uv pip install --reinstall "${LOCAL_WHEEL_PATH}"
elif [ -f "${UC_WHEEL_PATH}" ]; then
  echo "Installing wheel from Unity Catalog volume: ${UC_WHEEL_PATH}"
  uv pip install --reinstall "${UC_WHEEL_PATH}"
else
  echo "Error: Wheel not found at either location:"
  echo "  Local: ${LOCAL_WHEEL_PATH}"
  echo "  UC:    ${UC_WHEEL_PATH}"
  exit 1
fi

echo "aef-bng installation complete."
