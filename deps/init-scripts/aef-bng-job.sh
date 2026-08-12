#!/bin/bash
set -e

# --- Init script for DAB job clusters ---
# Installs uv and the aef-bng wheel from the bundle's artifact upload location.

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
echo "uv installed: $(uv --version)"

# Find the wheel uploaded by the DAB artifacts section.
# The DAB uploads wheels to: /Workspace/Users/<user>/.bundle/<bundle>/dev/artifacts/.internal/<name>.whl
# We search for it by glob pattern.
WHEEL_PATH=$(find /Workspace/Users -path "*/.bundle/aef-bng/*/artifacts/*" -name "aef_bng-*.whl" 2>/dev/null | head -n 1)

if [ -z "$WHEEL_PATH" ]; then
  # Fallback: check the local libs directory (where Databricks puts task libraries)
  WHEEL_PATH=$(find /local_disk0 /databricks/jars -name "aef_bng-*.whl" 2>/dev/null | head -n 1)
fi

if [ -z "$WHEEL_PATH" ]; then
  echo "Warning: aef_bng wheel not found in bundle artifacts. Falling back to pip install at task start."
  exit 0
fi

echo "Installing wheel via uv: ${WHEEL_PATH}"
uv pip install --system "${WHEEL_PATH}"
echo "aef-bng installation complete."
