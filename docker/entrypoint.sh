#!/bin/bash

# this entrypoint script does NOT include defaults. it expects env vars to be set by the container, 
# and should error out if they aren't

echo "Running entrypoint..."

# --- Configuration ---
# Directory mounted from host, whose ownership we need to match primarily
TARGET_DIR="${CONTROLLER_DIR}"
CEREBELLUM_PATH="${CEREBELLUM_PATH}"
SHARED_DATA_DIR="${SHARED_DATA_DIR}"
USERNAME="${USERNAME}"
VENV_PATH="${VIRTUAL_ENV}"
NEST_MODULE_PATH="${NEST_MODULE_PATH}"
COMPRESSED_BSB_NETWORK_FILE="${COMPRESSED_BSB_NETWORK_FILE}"
BSB_NETWORK_FILE="${BSB_NETWORK_FILE}"

PYTHON_MAJOR_MINOR=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
SITE_PACKAGES_PATH="$VENV_PATH/lib/${PYTHON_MAJOR_MINOR}/site-packages"

# --- UID/GID Synchronization ---
# CRITICAL CHECK: Ensure the target directory (bind mount) exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR not found or not a directory." >&2
    echo "This directory must be bind-mounted from the host." >&2
    exit 1
fi

# Get UID and GID of the target directory (mounted from host)
DIR_UID=$(stat -c "%u" "$TARGET_DIR")
DIR_GID=$(stat -c "%g" "$TARGET_DIR")
echo "Mounted directory $TARGET_DIR owned by UID: $DIR_UID, GID: $DIR_GID"

# Get current container user's UID and GID
CURRENT_UID=$(id -u "$USERNAME")
CURRENT_GID=$(id -g "$USERNAME")

# If UID/GID don't match the directory, change the container user's UID/GID
if [ "$CURRENT_UID" != "$DIR_UID" ] || [ "$CURRENT_GID" != "$DIR_GID" ]; then
    echo "Current $USERNAME UID/GID ($CURRENT_UID/$CURRENT_GID) differs from target ($DIR_UID/$DIR_GID). Adjusting..."

    # Ensure the target GID exists or modify the existing group
    if ! getent group "$DIR_GID" > /dev/null; then
        echo "Modifying group $USERNAME to GID $DIR_GID..."
        groupmod -o -g "$DIR_GID" "$USERNAME" # Allow duplicate GID, modify existing group
    else
        EXISTING_GROUP_NAME=$(getent group "$DIR_GID" | cut -d: -f1)
        if [ "$EXISTING_GROUP_NAME" != "$USERNAME" ]; then
             echo "Target GID $DIR_GID exists with name $EXISTING_GROUP_NAME. Modifying $USERNAME's primary group GID to $DIR_GID."
             # Check if user is already part of the target group, if so, make it primary
             if id -G "$USERNAME" | grep -qw "$DIR_GID"; then
                 usermod -g "$DIR_GID" "$USERNAME"
             else
                 # If target GID exists but belongs to another group, modify our user's group GID
                 groupmod -o -g "$DIR_GID" "$USERNAME"
             fi
        fi
    fi

    # Modify User: Change the user's UID
    echo "Modifying user $USERNAME to UID $DIR_UID..."
    usermod -o -u "$DIR_UID" "$USERNAME" # Allow duplicate UID

    # Adjust ownership of internal directories needed by the user
    echo "Adjusting ownership of internal directories ($VENV_PATH, /home/$USERNAME)..."
    chown -R "$DIR_UID:$DIR_GID" "$VENV_PATH" "/home/$USERNAME" "$SHARED_DATA_DIR" "$NEST_MODULE_PATH"

    echo "$USERNAME user adjusted to UID: $DIR_UID, GID: $DIR_GID"
else
    echo "$USERNAME UID/GID ($CURRENT_UID/$CURRENT_GID) matches target ($DIR_UID/$DIR_GID). No changes needed."
fi

# --- Decompress BSB Network File if necessary ---
echo "Checking for BSB network file: ${BSB_NETWORK_FILE}"
if [ ! -f "${BSB_NETWORK_FILE}" ]; then
    echo "Uncompressed network file ${BSB_NETWORK_FILE} not found."
    mkdir -p "$(dirname "${BSB_NETWORK_FILE}")" # Ensure parent directory exists
    echo "Found compressed file ${COMPRESSED_BSB_NETWORK_FILE}. Decompressing..."
    gzip -d -c "${COMPRESSED_BSB_NETWORK_FILE}" > "${BSB_NETWORK_FILE}"
else
    echo "Uncompressed network file ${BSB_NETWORK_FILE} already exists. Skipping decompression."
fi

echo "Installing custom stdp"
cd $CONTROLLER_DIR/built_custom_stdp
make install
cd $CONTROLLER_DIR

# --- Prerequisite Scripts ---
# Run the editable install script
echo "Running python dependencies script as user '$USERNAME'..."
export VIRTUAL_ENV CONTROLLER_DIR USERNAME BULLET_MUSCLE_DIR
gosu "$USERNAME" /usr/local/bin/prepare_python_env.sh


# Start VNC in the background AS THE USER first using the dedicated script.
echo "Entrypoint: Launching VNC background process via gosu..."
# Export variables needed by the background script
export VNC_DISPLAY VNC_PASSWORD HOME=/home/$USERNAME
gosu "$USERNAME" /usr/local/bin/start-vnc.sh

echo "Entrypoint: Executing custom command as user '$USERNAME': $@"

echo "----------------------------------------"
echo "Switching to user $USERNAME (UID: $DIR_UID, GID: $DIR_GID) and executing command: $@"
echo "----------------------------------------"

# --- Set Environment Variables for Final Command ---
# Ensure these are set *before* gosu executes the final command
# so they are inherited by the user's environment.

echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Final PATH: $PATH"
echo "Final PYTHONPATH: $PYTHONPATH"

exec gosu "$USERNAME" "$@"
# exec gosu "$USERNAME" bash -c 'run_as_user "$@"' bash "$@"
# python controller/complete_control/brain.py
# bash --rcfile <(python controller/complete_control/brain.py)