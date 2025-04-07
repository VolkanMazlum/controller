#!/bin/bash

# this entrypoint script does NOT include defaults. it expects env vars to be set by the container, 
# and should error out if they aren't

# export PYTHONPATH="/sim/nest-install/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
# export PATH="/sim/nest-install:/sim/controller:/sim/bullet_muscle_sim:${PATH}"
echo "Running setup to ensure permissions of mounted volume match with docker user UID/GID"

export PYTHONPATH="$PYTHONPATH:/sim/controller/cerebellum/"

# --- Configuration ---
# Directory mounted from host, whose ownership we need to match primarily
TARGET_DIR="${CONTROLLER_DIR}"
# Path to the source code inside container
CEREBELLUM_PATH_IN_CONTAINER="${TARGET_DIR}/cerebellum" # Assumes cerebellum is inside controller mount
SHARED_DATA_DIR_IN_CONTAINER="${SHARED_DATA_DIR}"
# User details in the container
USERNAME="${USERNAME}"
# Venv details
VENV_PATH="${VIRTUAL_ENV}"
NEST_MODULE_PATH="${NEST_MODULE_PATH}"

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
    chown -R "$DIR_UID:$DIR_GID" "$VENV_PATH" "/home/$USERNAME" "$SHARED_DATA_DIR_IN_CONTAINER" "$NEST_MODULE_PATH"

    echo "$USERNAME user adjusted to UID: $DIR_UID, GID: $DIR_GID"
else
    echo "$USERNAME UID/GID ($CURRENT_UID/$CURRENT_GID) matches target ($DIR_UID/$DIR_GID). No changes needed."
fi


# --- Editable Install Logic (runs AS the target user) ---
run_as_user() {
  # Variables needed inside this function (re-declared for clarity, but inherited via export)
  local cerebellum_path="$CEREBELLUM_PATH_IN_CONTAINER"
  local site_packages="$SITE_PACKAGES_PATH"
  local venv_path="$VENV_PATH"
  local shared_data_dir="$SHARED_DATA_DIR_IN_CONTAINER"

  echo "Running as user: $(id)"
  echo "Checking for cerebellum at: $cerebellum_path"
  echo "Checking site-packages at: $site_packages"

  # CRITICAL CHECK: Ensure the cerebellum source directory exists inside the container mount
  if [ ! -d "$cerebellum_path" ]; then
      echo "Error: Cerebellum source directory not found at '$cerebellum_path'." >&2
      echo "Editable install cannot proceed. Ensure it's correctly mounted/placed within $TARGET_DIR." >&2
      exit 1
  fi

  # check if cerebellum already installed
  local pth_file
  pth_file=$(find "$site_packages" -maxdepth 1 -name 'cerebellum*' -print -quit)

  if [ -z "$pth_file" ]; then
    echo "Editable package link for cerebellum not found in $site_packages. Installing..."
    # Use the venv pip directly. --no-cache-dir avoids potential cache permission issues.
    "$venv_path/bin/pip" install --no-cache-dir -e "$cerebellum_path"
    echo "Editable install of cerebellum complete."
  else
    echo "Editable package link for cerebellum found ($pth_file). Skipping install."
  fi

  # Execute the main container command passed via docker run or CMD
  echo "----------------------------------------"
  echo "Executing command as $(id -u -n): $@"
  echo "----------------------------------------"
  exec "$@"
}

# --- Execution ---
# Export variables needed by the run_as_user function in the gosu subshell
export CEREBELLUM_PATH_IN_CONTAINER VENV_PATH SITE_PACKAGES_PATH TARGET_DIR SHARED_DATA_DIR_IN_CONTAINER

# Export the function so the subshell run by gosu can find it
export -f run_as_user

# Use gosu to drop root privileges and switch to the (potentially modified) USERNAME
# Execute the run_as_user function defined above, passing along any arguments ($@)

exec gosu "$USERNAME" bash -c 'run_as_user "$@"' bash "$@"
# python controller/complete_control/brain.py
# bash --rcfile <(python controller/complete_control/brain.py)
