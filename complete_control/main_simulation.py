#!/usr/bin/env python3

import datetime
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

import config.paths as paths
import nest
import numpy as np
import structlog
from config.paths import RunPaths, setup_run_paths
from config.settings import SEED, Experiment, Simulation
from mpi4py import MPI
from mpi4py.MPI import Comm
from neural.Controller import Controller
from neural.data_handling import collapse_files
from neural.plot_utils import plot_controller_outputs
from utils_common.generate_analog_signals import generate_signals
from utils_common.log import setup_logging, tqdm

nest.set_verbosity("M_ERROR")  # M_WARNING


# --- Configuration and Setup ---
def load_config(json_path):
    log = structlog.get_logger("main.config")
    """Loads parameters from JSON file."""
    with open(json_path) as f:
        params = json.load(f)
        log.info("Configuration loaded", path=json_path)
    return params


def setup_environment(nestml_build_dir=paths.NESTML_BUILD_DIR):
    log = structlog.get_logger("main.env_setup")
    """Sets up environment variables if needed (e.g., for NESTML)."""
    # This is fragile. Better to manage environments or installation paths.
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    nestml_path_str = str(nestml_build_dir)
    if nestml_path_str not in ld_lib_path:
        # Check if the path exists before adding
        if nestml_build_dir.exists():
            new_path = (
                ld_lib_path + ":" + nestml_path_str if ld_lib_path else nestml_path_str
            )
            os.environ["LD_LIBRARY_PATH"] = new_path
            log.info("Updated LD_LIBRARY_PATH", path_added=nestml_path_str)
        else:
            log.warning(
                "NESTML path not found, skipping LD_LIBRARY_PATH update",
                path=nestml_path_str,
            )
    # Import NESTML models after path setup
    try:
        # Check if module is already installed to prevent errors on reset
        if "controller_module" not in nest.Models(mtype="nodes"):
            # TODO we should only install it if we are running without cerebellum, otherwise
            # cerebellum tries installing it again (or remove it from configuration)
            # nest.Install("controller_module")  # Install custom NESTML modules
            log.info("Installed NESTML module", module="controller_module")
        else:
            log.debug("NESTML module already installed", module="controller_module")
    except nest.NESTError as e:
        # Handle cases where installation fails even if not previously installed
        log.error(
            "Error installing NESTML module",
            module="controller_module",
            error=str(e),
            exc_info=True,
        )
        log.error(
            "Ensure module is compiled and accessible (check LD_LIBRARY_PATH/compilation)."
        )
        sys.exit(1)


# --- NEST Kernel Setup ---
def setup_nest_kernel(sim: Simulation, seed: int, path_data: Path):
    log = structlog.get_logger("main.nest_setup")
    """Configures the NEST kernel."""

    kernel_status = {
        "resolution": sim.resolution,
        "overwrite_files": True,  # optional since different data paths
        "data_path": str(path_data),
        # "print_time": True, # Optional: Print simulation progress
    }
    kernel_status["rng_seed"] = seed  # Set seed via kernel status
    nest.SetKernelStatus(kernel_status)
    log.info(
        f"NEST Kernel: Resolution: {sim.resolution}ms, Seed: {seed}, Data path: {str(path_data)}"
    )
    random.seed(seed)
    np.random.seed(seed)


# --- Simulation Execution ---
def run_simulation(
    sim: Simulation,
    n_trials: int,
    path_data: Path,
    controllers: list[Controller],
    comm: Comm,
):
    log: structlog.stdlib.BoundLogger = structlog.get_logger("main.simulation_loop")
    """Runs the NEST simulation for the specified number of trials."""
    single_trial_ms = sim.duration_single_trial_ms

    # --- Prepare for Data Collapsing ---
    pop_views_to_collapse_by_label = defaultdict(list)
    for controller in controllers:
        for view in controller.pops.get_all_views():
            if view.label is not None:  # Check if label exists (meaning to_file=True)
                pop_views_to_collapse_by_label[view.label].append(view)

    # Extract unique labels and the grouped PopView lists
    unique_labels = list(pop_views_to_collapse_by_label.keys())
    grouped_pop_views = list(pop_views_to_collapse_by_label.values())

    if unique_labels:
        log.info(
            f"Found {len(unique_labels)} population types for data collapsing based on labels: {unique_labels}"
        )
    else:
        log.info(
            "No populations configured for file output (to_file=True with a label)."
        )

    for trial in range(n_trials):
        current_sim_start_time = nest.GetKernelStatus("biological_time")
        log.info(
            f"Starting Trial {trial + 1}/{n_trials}",
            duration_ms=total_sim_time_ms,
            current_sim_time_ms=current_sim_start_time,
        )
        log.info(f"Current simulation time: {current_sim_start_time} ms")
        start_trial_time = timer()

        nest.Simulate(total_sim_time_ms)

        end_trial_time = timer()
        trial_wall_time = timedelta(seconds=end_trial_time - start_trial_time)
        log.info(
            f"Finished Trial {trial + 1}/{n_trials}",
            sim_time_end_ms=nest.GetKernelStatus("biological_time"),
            wall_time=str(trial_wall_time),
        )

        # --- Data Collapsing ---
        log.info("Attempting data collapsing...")
        start_collapse_time = timer()
        if grouped_pop_views:
            collapse_files(
                str(path_data) + "/",
                unique_labels,
                grouped_pop_views,
                len(controllers),
                comm,
            )
            log.info("Data collapsing function executed.")
        else:
            log.info("No populations configured for file output found to collapse.")

        end_collapse_time = timer()
        collapse_wall_time = timedelta(seconds=end_collapse_time - start_collapse_time)
        log.info(
            f"Trial {trial + 1} data collapsing finished",
            wall_time=str(collapse_wall_time),
        )

    log.info("--- Simulation Finished ---")


def coordinate_paths_with_receiver() -> tuple[str, RunPaths]:
    shared_data = {
        "timestamp": None,
        "paths": None,
    }
    run_timestamp_str = None
    if rank == 0:
        shared_data["timestamp"] = run_timestamp_str = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        shared_data["paths"] = setup_run_paths(run_timestamp_str)
        print("sending paths to all processes...")

    shared_data = MPI.COMM_WORLD.bcast(shared_data, root=0)
    run_timestamp_str = shared_data["timestamp"]
    run_paths: RunPaths = shared_data["paths"]

    return run_timestamp_str, run_paths


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- MPI Setup ---
    comm = MPI.COMM_WORLD.Create_group(  # last process is for receiver_plant
        MPI.COMM_WORLD.group.Excl([MPI.COMM_WORLD.Get_size() - 1])
    )
    rank = comm.rank
    run_timestamp_str, run_paths = coordinate_paths_with_receiver()
    setup_logging(
        MPI.COMM_WORLD,
        log_dir_path=run_paths.logs,
        timestamp_str=run_timestamp_str,
        log_level=os.environ.get("LOG_LEVEL", "DEBUG"),
    )

    main_log: structlog.stdlib.BoundLogger = structlog.get_logger("main")
    main_log.info(
        f"Starting Run: {run_timestamp_str}",
        run_dir=str(run_paths.run),
        log_all_ranks=True,
    )
    main_log.info(
        "MPI Setup Complete",
        world_rank=MPI.COMM_WORLD.Get_rank(),
        world_size=MPI.COMM_WORLD.Get_size(),
        sim_rank=comm.rank,
        sim_size=comm.size,
        log_all_ranks=True,
    )
    start_script_time = timer()
    nest.ResetKernel()
    config_source = paths.PARAMS

    params = load_config(config_source)
    sim = Simulation()
    exp = Experiment()

    module_params = params["modules"]
    pops_params = params["pops"]
    conn_params = params["connections"]
    main_log.debug(
        "Loaded Parameters", params=params, sim_settings=sim, exp_settings=exp
    )
    # --- Copy Config to Run Directory (Rank 0 only) ---
    if rank == 0:
        try:
            config_dest = run_paths.run / Path(config_source).name
            shutil.copy2(config_source, config_dest)  # copy2 preserves metadata
            main_log.info(
                "Copied config file to run directory",
                source=config_source,
                destination=str(config_dest),
            )
        except FileNotFoundError:
            main_log.warning("Config file not found, skipping copy", path=config_source)
        except Exception as e:
            main_log.error("Error copying config file", error=str(e), exc_info=True)

    # TODO: Consider adding an "experiment" section to new_params.json
    exp_params = {
        "seed": SEED,  # Generate random seed if not provided
        "N": 50,  # Default number of neurons per sub-population
        # Add other experiment-level info if needed
    }
    seed = exp_params["seed"]
    N = exp_params["N"]

    # main_log.info("Simulation Parameters", sim)
    main_log.info("Experiment Parameters", **exp_params)

    # TODO: Consider adding a "music" section to new_params.json
    music_cfg = {
        "out_port": "mot_cmd_out",
        "in_port": "fbk_in",
        "const": 1e-6,  # Latency constant
    }
    main_log.info("MUSIC Configuration", **music_cfg)

    setup_environment()

    trj, motor_commands = generate_signals()

    njt = 1
    main_log.info(f"assuming {njt} DoF.")

    main_log.info("Input data loaded", dof=njt)
    # --- Time Vectors ---
    res = sim.resolution
    time_span_per_trial = sim.duration_single_trial_ms
    n_trials = sim.n_trials
    total_sim_duration = time_span_per_trial * n_trials

    # Time vector for a single trial (passed to PopView)
    single_trial_time_vect = np.linspace(
        0,
        time_span_per_trial,
        num=int(np.round(time_span_per_trial / res)),
        endpoint=True,
    )
    # Total time vector across all trials (for plotting concatenated results)
    total_time_vect_concat = np.linspace(
        0,
        total_sim_duration,
        num=int(np.round(total_sim_duration / res)),
        endpoint=True,
    )

    main_log.debug(
        "Time vectors calculated",
        total_duration=total_sim_duration,
        single_trial_duration=time_span_per_trial,
        num_steps_total=len(total_time_vect_concat),
        num_steps_trial=len(single_trial_time_vect),
    )

    # --- Network Construction ---
    start_network_time = timer()
    setup_nest_kernel(sim, seed, run_paths.data_nest)

    controllers = []
    main_log.info(f"Constructing Network", dof=njt, N=N)
    for j in range(njt):
        main_log.info(f"Creating controller", dof=j)
        mc_p = module_params["motor_cortex"]
        plan_p = module_params["planner"]
        spine_p = module_params["spine"]
        state_p = module_params["state"]

        controller = Controller(
            dof_id=j,
            N=N,
            total_time_vect=single_trial_time_vect,
            trajectory_slice=trj,
            motor_cmd_slice=motor_commands,
            mc_params=mc_p,
            plan_params=plan_p,
            spine_params=spine_p,
            state_params=state_p,
            pops_params=pops_params,
            conn_params=conn_params,
            sim_params=sim,
            path_data=run_paths.data_nest,
            label_prefix="",
            music_cfg=music_cfg,
            use_cerebellum=True,
            cerebellum_config={},  # TODO
            comm=comm,
        )
        controllers.append(controller)

    # --- Inter-Controller Connections (if any) ---
    # Add code here if controllers need to be connected to each other

    end_network_time = timer()
    main_log.info(
        f"Network Construction Finished",
        wall_time=str(timedelta(seconds=end_network_time - start_network_time)),
    )

    # --- Simulation ---
    run_simulation(sim_params, n_trials, run_paths.data_nest, controllers, comm)

    # --- Plotting (Rank 0 Only) ---
    if rank == 0:
        main_log.info("--- Generating Plots ---")
        start_plot_time = timer()
        try:
            plot_controller_outputs(
                controllers, total_time_vect_concat, run_paths.figures
            )
        except Exception as e:
            main_log.error("Error during plotting", error=str(e), exc_info=True)
        end_plot_time = timer()
        plot_wall_time = timedelta(seconds=end_plot_time - start_plot_time)
        main_log.info(f"Plotting Finished", wall_time=str(plot_wall_time))

    # --- Final Timing ---
    end_script_time = timer()
    main_log.info(f"--- Script Finished ---")
    main_log.info(
        f"Total wall clock time: {timedelta(seconds=end_script_time - start_script_time)}"
    )
