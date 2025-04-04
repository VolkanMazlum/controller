#!/bin/bash
set -euo pipefail # Exit on error, unset var, pipe failure

# --- Configuration & Validation ---

# Build directory (absolute path)
BUILD_DIR="/sim/controller/built_custom_stdp"

# Check required environment variables
: "${CONTROLLER_DIR:?Error: CONTROLLER_DIR environment variable is not set}"
: "${NEST_INSTALL_DIR:?Error: NEST_INSTALL_DIR environment variable is not set (needed for cmake)}"

# Source directory for the custom module
SOURCE_DIR="${CONTROLLER_DIR}/cerebellum/custom_stdp"

# Validate that the source directory exists
if [ ! -d "${SOURCE_DIR}" ]; then
   echo "Error: Source directory for custom_stdp not found: ${SOURCE_DIR}" >&2
   exit 1
fi

# --- Preparation ---
echo "Ensuring build directory exists: ${BUILD_DIR}"
# Create the build directory if it doesn't exist. Does nothing if it exists.
mkdir -p "${BUILD_DIR}" || { echo "Error: Failed to create directory ${BUILD_DIR}" >&2; exit 1; }

# --- Check Contents & Confirmation ---

# Check if the directory contains any files or directories (excluding . and ..)
# The '-A' flag includes hidden files/dirs except . and ..
NEEDS_CLEARING=false
if [ -n "$(ls -A "${BUILD_DIR}")" ]; then
    NEEDS_CLEARING=true
    echo "---------------------------------------------------------------------"
    echo "Warning: Build directory '${BUILD_DIR}' is not empty."
    # Use read with -r to handle backslashes, -p for prompt, default to 'n'
    read -r -p "Do you want to clear its contents and proceed? (y/N): " confirm
    if [[ "$(echo "${confirm:-n}" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
        echo "Aborting regeneration as clearing was not confirmed."
        echo "---------------------------------------------------------------------"
        exit 0 # Graceful exit
    fi
    echo "---------------------------------------------------------------------"
fi

# --- Clear Directory (if needed) ---
if [ "$NEEDS_CLEARING" = true ]; then
    echo "Clearing contents of ${BUILD_DIR}..."
    # Use find to delete all contents (files, directories, hidden) within the directory
    find "${BUILD_DIR}" -mindepth 1 -delete || { echo "Error: Failed to clear contents of ${BUILD_DIR}" >&2; exit 1; }
    echo "Directory contents cleared."
fi

# --- Build ---
NEST_BIN=$NEST_INSTALL_DIR/bin/nest-config
echo "---------------------------------------------------------------------"
echo "Starting CMake, Make, and Make Install process..."
echo "Build directory: ${BUILD_DIR}"
echo "Source directory: ${SOURCE_DIR}"
echo "NEST binary: ${NEST_BIN}"
echo "---------------------------------------------------------------------"

# Change into the build directory
echo "Changing directory to ${BUILD_DIR}"
cd "${BUILD_DIR}" || exit 1 # Exit if cd fails

# Run CMake
echo "Running CMake..."
cmake -Dwith-nest="${NEST_BIN}" "${SOURCE_DIR}"

# Run Make
echo "Running make..."
make

# Run Make Install
echo "Running make install..."
make install

# --- Completion Message ---
# set -e handles errors, so if we reach here, it succeeded.
echo "---------------------------------------------------------------------"
echo "Custom STDP module regenerated and installed successfully."
echo "Build directory: ${BUILD_DIR}"
echo "---------------------------------------------------------------------"

exit 0