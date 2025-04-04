# what's this?
this is where i'll store notes about the various scripts that make this whole thing possible

bsb compile -v4 -o "/sim/controller/built_models/from_microzones_complete_nest@$(date +%F_%T).hdf5" configurations/mouse/dcn-io/microzones_complete_nest.yaml | tee -a /sim/controller/logs/$(date +%F_%T).txt


# what is the intended way this would work?
here's my mid-term plan... which may never be implemented:

Build: The Dockerfile creates /sim/shared_data, sets its initial ownership (based on --build-args), copies run_interactive.py, and sets the default CMD to run this script.

Run (First Time):
    docker-compose up --build (or run --rm) starts the container.
    Docker creates two named volumes: sim_venv and sim_shared_data.
    The entrypoint runs, matches UID/GID, and importantly chowns /sim/venv, /home/simuser, and /sim/shared_data to the correct user.
    The entrypoint performs the editable install of cerebellum (into sim_venv).
    The entrypoint finally executes the default command: python /usr/local/bin/run_interactive.py.
    The Python script checks /sim/shared_data. Since it's empty, it determines regeneration is needed for both steps.
    It runs your (placeholder) network compilation command, saving output to /sim/shared_data/network_description.h5.
    It runs your (placeholder) trajectory generation command, saving output to /sim/shared_data/trajectory_data/.
    It asks if you want to run the simulation. If yes, it runs the main simulation command using the files in /sim/shared_data.
Run (Subsequent Times):
    docker-compose run --rm simulation starts a new container.
    Docker reuses the existing sim_venv and sim_shared_data volumes.
    The entrypoint runs, fixes permissions again (quick operation), finds the cerebellum .egg-link in sim_venv and skips the install.
    It executes python /usr/local/bin/run_interactive.py.
    The Python script finds network_description.h5 and the trajectory_data directory in /sim/shared_data.
    It asks you: "Regenerate network description? (y/N)". If you press Enter (or 'n'), it skips regeneration.
    It asks you: "Regenerate trajectory data? (y/N)". If you press Enter (or 'n'), it skips regeneration.
    It asks: "Proceed to run the main simulation? (Y/n)". If you press Enter (or 'y'), it runs the simulation using the existing files from the volume.