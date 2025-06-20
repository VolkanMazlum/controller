#!/bin/bash
set -e

# This script runs as the non-root user ($USERNAME) after gosu.
# It ensures the 'cerebellum' package is installed editable.

# --- Inherited Environment Variables ---
# VIRTUAL_ENV: Path to the virtual environment
# CONTROLLER_DIR: Path to the controller directory mount (/sim/controller)
# USERNAME: The non-root user name (e.g., simuser)
# CEREBELLUM_PATH

# --- Configuration ---
REQUIREMENTS_PATH="${CONTROLLER_DIR}/docker/requirements.txt"
PYTHON_MAJOR_MINOR=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES_PATH="$VIRTUAL_ENV/lib/${PYTHON_MAJOR_MINOR}/site-packages"

echo "Running editable install setup as user: $(id)"
echo "Virtual Env: $VIRTUAL_ENV"
echo "Controller Dir: $CONTROLLER_DIR"
echo "Site Packages: $SITE_PACKAGES_PATH"

# Ensure the primary requirements from the mounted dir are installed
if [ -f "$REQUIREMENTS_PATH" ]; then
  echo "Installing requirements specified in $REQUIREMENTS_PATH..."
  "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -qr "$REQUIREMENTS_PATH"
else
  echo "Warning: Application requirements file not found at $REQUIREMENTS_PATH" >&2
  exit 1
fi

# Ensure the cerebellum source directory exists inside the container mount
if [ ! -d "$CEREBELLUM_PATH" ]; then
    echo "Warning: Cerebellum source directory not found at '$CEREBELLUM_PATH'." >&2
    echo "Editable install cannot proceed. Ensure it's correctly mounted/placed within $CONTROLLER_DIR." >&2
    exit 1
else
    # check if cerebellum already installed as editable
    pth_file=$(find "$SITE_PACKAGES_PATH" -maxdepth 1 -name 'cerebellum*' -print -quit)

    # Check if cerebellum is installed OR if bsb-core version is not 5.1.1
    bsb_core_check_command="\"$VIRTUAL_ENV/bin/pip\" show bsb-core"
    # Check if the output of pip show contains the version string "5.1.1"
    if [ -z "$pth_file" ] || ! $bsb_core_check_command | grep -q "5.1.1"; then
        if [ -z "$pth_file" ]; then
            echo "Editable package link for cerebellum not found in $SITE_PACKAGES_PATH."
        else
            # Get the actual version for logging purposes
            actual_version=$($bsb_core_check_command | grep '^Version:' | awk '{print $2}')
            echo "bsb-core version is not 5.1.1 (found: $actual_version)."
        fi
        echo "Running TEMPORARY PATCH to ensure correct dependencies..."


        # TEMPORARY PATCH START >>>>>>>>>>>>
        echo "Making an absolute mess of the bsb dependencies to get a working environment.."
        # first install cerebellum@feat/plasticity, which will install bsb@4.1
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -e "$CEREBELLUM_PATH"
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q bsb-core@git+https://github.com/dbbs-lab/bsb-core.git@dev
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q bsb-nest@git+https://github.com/dbbs-lab/bsb-nest.git@dev
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q bsb-hdf5@git+https://github.com/dbbs-lab/bsb-hdf5.git@dev
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q bsb-json==4.2.3
        "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q bsb-yaml==4.2.3
        # ============================================
        # "$VIRTUAL_ENV/bin/pip" install --no-cache-dir -q -e "$CEREBELLUM_PATH"
        # TEMPORARY PATCH END <<<<<<<<<<<<<<<<<<<<<<<
        echo "TEMPORARY PATCH execution complete."
    else
        echo "Editable package link for cerebellum found ($pth_file) and bsb-core version appears to be 5.1.1. Skipping patch."
    fi
fi
# Environment variables are now set in entrypoint.sh before the final gosu exec

echo "Editable install check finished."