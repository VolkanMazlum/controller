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

import nest
import numpy as np
import paths
import structlog
from data_handling import collapse_files
from log import setup_logging, tqdm
from mpi4py import MPI
from mpi4py.MPI import Comm
from paths import RunPaths, setup_run_paths
from plot_utils import plot_controller_outputs
from settings import SEED, Experiment, Simulation

from controller import SingleDOFController


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


def load_input_data(traj_file: Path, cmd_file: Path):
    log = structlog.get_logger("main.input_data")
    """Loads trajectory and motor command data."""
    try:
        trj = np.loadtxt(traj_file)
        motor_commands = np.loadtxt(cmd_file)

        if trj.shape[0] != motor_commands.shape[0]:
            log.error(
                "Input data length mismatch",
                traj_file=traj_file,
                traj_len=trj.shape[0],
                cmd_file=cmd_file,
                cmd_len=motor_commands.shape[0],
            )
            raise ValueError(
                f"Trajectory ({traj_file}, len {trj.shape[0]}) and motor command ({cmd_file}, len {motor_commands.shape[0]}) files have different lengths."
            )
        # Handle 1D vs 2D data consistently
        if trj.ndim == 1:
            trj = trj.reshape(-1, 1)
        if motor_commands.ndim == 1:
            motor_commands = motor_commands.reshape(-1, 1)
        log.info(
            "Loaded input data",
            trajectory_shape=trj.shape,
            motor_cmd_shape=motor_commands.shape,
        )
        return trj, motor_commands
    except FileNotFoundError as e:
        log.error(
            f"Error loading input data: {e}. Ensure '{traj_file}' and '{cmd_file}' exist."
        )
        sys.exit(1)
    except ValueError as e:
        log.error(f"Error processing input data files: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        log.error(
            f"An unexpected error occurred during input data loading: {e}",
            exc_info=True,
        )
        sys.exit(1)


# --- NEST Kernel Setup ---
def setup_nest_kernel(sim_params: dict, seed: int, path_data: Path):
    log = structlog.get_logger("main.nest_setup")
    """Configures the NEST kernel."""

    kernel_status = {
        "resolution": sim_params["res"],
        "overwrite_files": True,  # Important for multiple trials/runs
        "data_path": str(path_data),
        # "print_time": True, # Optional: Print simulation progress
    }
    kernel_status["rng_seed"] = seed  # Set seed via kernel status
    nest.SetKernelStatus(kernel_status)
    log.info(
        f"NEST Kernel: Resolution: {sim_params['res']}ms, Seed: {seed}, Data path: {str(path_data)}"
    )
    random.seed(seed)
    np.random.seed(seed)


# --- MUSIC Setup ---
def setup_music_interface(n_dof: int, N: int, msc_params: dict, spine_params: dict):
    log = structlog.get_logger("main.music_setup")
    """Creates MUSIC proxies for input and output."""
    n_total_neurons = 2 * N * n_dof

    out_port_name = msc_params["out_port"]
    in_port_name = msc_params["in_port"]
    latency_const = msc_params["const"]

    # Output proxy
    log.info("Creating MUSIC out proxy", port=out_port_name)
    proxy_out = nest.Create(
        "music_event_out_proxy", 1, params={"port_name": out_port_name}
    )
    log.info("Created MUSIC out proxy", port=out_port_name, gids=proxy_out.tolist())

    # Input proxy
    proxy_in = nest.Create(
        "music_event_in_proxy", n_total_neurons, params={"port_name": in_port_name}
    )
    log.info("Creating MUSIC in proxy", port=in_port_name, channels=n_total_neurons)
    for i, n in enumerate(proxy_in):
        nest.SetStatus(n, {"music_channel": i})
    log.info(
        f"Created MUSIC in proxy: port '{in_port_name}' with {n_total_neurons} channels"
    )

    # We need to tell MUSIC, through NEST, that it's OK (due to the delay)
    # to deliver spikes a bit late. This is what makes the loop possible.
    # Set acceptable latency for the input port
    # Use feedback delay from spine parameters
    fbk_delay = spine_params["fbk_delay"]
    latency = fbk_delay - latency_const
    # if latency < nest.GetKernelStatus("min_delay"):
    #     print(
    #         f"Warning: Calculated MUSIC latency ({latency}) is less than min_delay ({nest.GetKernelStatus('min_delay')}). Clamping to min_delay."
    #     )
    #     latency = nest.GetKernelStatus("min_delay")

    nest.SetAcceptableLatency(in_port_name, latency)
    log.info("Set MUSIC acceptable latency", port=in_port_name, latency=latency)

    return proxy_in, proxy_out


def connect_controller_to_music(
    controller: SingleDOFController, proxy_in, proxy_out, dof_id, N
):
    log = structlog.get_logger(f"main.music_connect.dof_{dof_id}")
    """Connects a single controller's inputs/outputs to MUSIC proxies."""
    log.debug("Connecting MUSIC interfaces", N=N)  # dof_id is in logger name
    # Get PopView objects
    bs_p_view, bs_n_view = controller.get_brainstem_output_popviews()
    sn_p_view, sn_n_view = controller.get_sensory_input_popviews()
    # Connect Brainstem outputs (access .pop from PopView)
    if bs_p_view and bs_n_view:
        start_channel_out = 2 * N * dof_id
        log.debug(
            "Connecting brainstem outputs to MUSIC out proxy",
            start_channel=start_channel_out,
            num_neurons=N,
        )
        for i, neuron in enumerate(bs_p_view.pop):
            nest.Connect(
                neuron,
                proxy_out,
                "one_to_one",
                {"music_channel": start_channel_out + i},
            )
        for i, neuron in enumerate(bs_n_view.pop):
            nest.Connect(
                neuron,
                proxy_out,
                "one_to_one",
                {"music_channel": start_channel_out + N + i},
            )
    # Connect MUSIC In Proxy to Sensory Neuron inputs (access .pop from PopView)
    start_channel_in = 2 * N * dof_id
    idx_start_p = start_channel_in
    idx_end_p = idx_start_p + N
    idx_start_n = idx_end_p
    idx_end_n = idx_start_n + N
    delay = controller.spine_params["fbk_delay"]
    wgt = controller.spine_params["wgt_sensNeur_spine"]
    log.debug(
        "Connecting MUSIC in proxy to sensory inputs",
        start_channel=start_channel_in,
        num_neurons=N,
        delay=delay,
        weight=wgt,
    )
    nest.Connect(
        proxy_in[idx_start_p:idx_end_p],
        sn_p_view.pop,
        "one_to_one",
        {"weight": wgt, "delay": delay},
    )
    nest.Connect(
        proxy_in[idx_start_n:idx_end_n],
        sn_n_view.pop,
        "one_to_one",
        {"weight": wgt, "delay": delay},
    )


# --- Simulation Execution ---
def run_simulation(
    sim_params: dict,
    n_trials: int,
    path_data: Path,
    controllers: list[SingleDOFController],
    comm: Comm,
):
    log: structlog.stdlib.BoundLogger = structlog.get_logger("main.simulation_loop")
    """Runs the NEST simulation for the specified number of trials."""
    total_sim_time_ms = sim_params["timeMax"] + sim_params["timeWait"]

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
        print(f"Current simulation time: {current_sim_start_time} ms")
        start_trial_time = timer()

        nest.Simulate(total_sim_time_ms)

        end_trial_time = timer()
        trial_wall_time = timedelta(seconds=end_trial_time - start_trial_time)
        log.info(
            f"Finished Trial {trial + 1}/{n_trials}",
            sim_time_end_ms=nest.GetKernelStatus("biological_time"),
            wall_time=str(trial_wall_time),
        )

        log.debug("Checking spike events", population="fbk_smooth_p")
        senders_refactored, times_refactored = controller.pops.fbk_smooth_p.get_events()
        log.debug(f"--- Refactored fbk_smooth_p spikes ---")
        log.debug(f"Num spikes: {len(times_refactored)}")
        if len(times_refactored) > 0:
            print(
                f"First spike time: {times_refactored[0]}, sender: {senders_refactored[0]}"
            )
            print(
                f"Last spike time: {times_refactored[-1]}, sender: {senders_refactored[-1]}"
            )

        # --- Data Collapsing ---
        log.info("Attempting data collapsing...")
        start_collapse_time = timer()
        # Use the dynamically gathered labels and views for collapsing
        if grouped_pop_views:
            # collapse_files expects a string path ending with "/"
            data_path_str = str(path_data) + "/"
            collapse_files(
                data_path_str,
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
        print("sending...")

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
        comm,
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

    # --- Load Configuration ---
    params = load_config(config_source)
    sim = Simulation()
    exp = Experiment()

    # --- Extract Parameters or Set Defaults ---
    # Module, Pop, Connection params (directly from JSON)
    module_params = params.get("modules", {})
    pops_params = params.get("pops", {})
    conn_params = params.get("connections", {})
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

    # Simulation parameters (provide defaults if not in JSON)
    # TODO: Consider adding a "simulation" section to new_params.json
    sim_params = {
        "res": sim.resolution,  # ms - Simulation resolution
        "timeMax": sim.timeMax,  # ms - Duration of one trial segment
        "timeWait": sim.timeWait,  # ms - Pause/wait time after trial segment
        "n_trials": sim.n_trials,  # Number of trials to run
        "clear_old_data": True,  # Option to clear previous data
    }

    # Experiment parameters (provide defaults if not in JSON)
    # TODO: Consider adding an "experiment" section to new_params.json
    exp_params = {
        "seed": SEED,  # Generate random seed if not provided
        "N": 50,  # Default number of neurons per sub-population
        # Add other experiment-level info if needed
    }
    seed = exp_params["seed"]
    N = exp_params["N"]

    main_log.info("Simulation Parameters", **sim_params)
    main_log.info("Experiment Parameters", **exp_params)

    # MUSIC configuration (provide defaults if not in JSON)
    # TODO: Consider adding a "music" section to new_params.json
    music_cfg = {
        "out_port": "mot_cmd_out",
        "in_port": "fbk_in",
        "const": 1e-6,  # Latency constant
    }
    main_log.info("MUSIC Configuration", **music_cfg)

    # --- Environment Setup ---
    setup_environment()
    # --- Load Input Data ---
    trj, motor_commands = load_input_data(paths.TRAJECTORY, paths.MOTOR_COMMANDS)
    njt = trj.shape[1]  # Infer number of DoFs
    main_log.info(f"Inferred {njt} DoF(s) from input data.")

    main_log.info("Input data loaded", dof=njt)
    # --- Time Vectors ---
    res = sim_params["res"]
    time_span_per_trial = sim_params["timeMax"] + sim_params["timeWait"]
    n_trials = sim_params["n_trials"]
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
    # Verify data length against single trial time vector
    if len(trj) != len(single_trial_time_vect):
        main_log.warning(
            f"Input data length ({len(trj)}) does not match single trial time vector length ({len(single_trial_time_vect)} based on timeMax={sim_params['timeMax']}, timeWait={sim_params['timeWait']}, res={res})."
        )
        # Option 1: Truncate data (might lose information)
        # if len(trj) > len(single_trial_time_vect):
        #     trj = trj[:len(single_trial_time_vect)]
        #     motor_commands = motor_commands[:len(single_trial_time_vect)]
        #     print("Truncated input data to match single trial duration.")
        # Option 2: Adjust simulation times (might not be desired)
        # Option 3: Assume data spans *all* trials (requires different handling in Controller) - Less likely based on original code
        main_log.warning(
            "Proceeding with potentially mismatched data/time vector length. Check parameters."
        )

    # --- Network Construction ---
    start_network_time = timer()
    setup_nest_kernel(sim_params, seed, run_paths.data_nest)

    controllers = []
    main_log.info(f"Constructing Network", dof=njt, N=N)
    for j in range(njt):
        main_log.info(f"Creating controller", dof=j)
        mc_p = module_params["motor_cortex"]
        plan_p = module_params["planner"]
        spine_p = module_params["spine"]
        state_p = module_params["state"]

        controller = SingleDOFController(
            dof_id=j,
            N=N,
            total_time_vect=single_trial_time_vect,  # Pass single trial vector
            trajectory_slice=trj[:, j],
            motor_cmd_slice=motor_commands[:, j],
            mc_params=mc_p,
            plan_params=plan_p,
            spine_params=spine_p,
            state_params=state_p,
            pops_params=pops_params,
            conn_params=conn_params,
            sim_params=sim_params,
            path_data=run_paths.data_nest,
            label_prefix="",
            use_cerebellum=True,
            cerebellum_config={},  # TODO
            comm=comm,
        )
        controllers.append(controller)

    # --- MUSIC Setup and Connection ---
    spine_params = module_params["spine"]  # Get spine params again for MUSIC setup
    proxy_in, proxy_out = setup_music_interface(njt, N, music_cfg, spine_params)
    for j, controller in enumerate(controllers):
        main_log.info(f"Connecting controller to MUSIC", dof=j)
        connect_controller_to_music(controller, proxy_in, proxy_out, j, N)

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
