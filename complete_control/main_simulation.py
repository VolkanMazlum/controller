#!/usr/bin/env python3

import json
import os
import random  # Import random for seed setting
import sys
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt

# NEST and MPI related imports
import nest
import numpy as np
from data_handling import collapse_files
from mpi4py import MPI
from settings import SEED, Experiment, Simulation

# --- Local Imports ---
# Assuming these are in the same directory or Python path
from controller import SingleDOFController  # Import the updated controller
from population_view import plotPopulation  # Import plotting function

# Import plotting functions if they are moved to a separate file
# from plot_utils import plot_controller_outputs


# --- Configuration and Setup ---
def load_config(json_path="new_params.json"):
    """Loads parameters from JSON file."""
    with open(json_path) as f:
        params = json.load(f)
        print(f"Successfully loaded configuration from '{json_path}'")
    return params


def setup_environment():
    """Sets up environment variables if needed (e.g., for NESTML)."""
    # This is fragile. Better to manage environments or installation paths.
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    nestml_path = (
        "../nestml/target"  # Consider making this configurable via JSON or env var
    )
    if nestml_path not in ld_lib_path:
        # Check if the path exists before adding
        if Path(nestml_path).exists():
            new_path = ld_lib_path + ":" + nestml_path if ld_lib_path else nestml_path
            os.environ["LD_LIBRARY_PATH"] = new_path
            print(f"Updated LD_LIBRARY_PATH to include: {nestml_path}")
        else:
            print(
                f"Warning: NESTML path '{nestml_path}' not found. Skipping LD_LIBRARY_PATH update."
            )

    # Import NESTML models after path setup
    try:
        # Check if module is already installed to prevent errors on reset
        if "controller_module" not in nest.Models(mtype="nodes"):
            nest.Install("controller_module")  # Install custom NESTML modules
            print("Installed NESTML module: controller_module")
        else:
            print("NESTML module 'controller_module' already installed.")
    except nest.NESTError as e:
        # Handle cases where installation fails even if not previously installed
        print(f"Error installing NESTML module 'controller_module': {e}")
        print(
            "Ensure the module is compiled and accessible (check LD_LIBRARY_PATH and compilation in '../nestml/target')."
        )
        sys.exit(1)


def setup_paths(fig_base_path_str: str):
    """Creates output directories based on figure path from config."""
    fig_base_path = Path(fig_base_path_str)
    # Define data path relative to figure path or make it configurable
    path_data = Path(".") / "data" / "nest"  # Example: place data dir alongside fig dir
    path_fig = fig_base_path

    path_data.mkdir(parents=True, exist_ok=True)
    path_fig.mkdir(parents=True, exist_ok=True)
    print(f"Using Figure Path: {path_fig}")
    print(f"Using Data Path:   {path_data}")
    return str(path_data), str(path_fig)


def clear_old_data(path_data_str: str):
    """Removes old simulation files in the specified NEST data path."""
    path_data = Path(path_data_str)
    if MPI.COMM_WORLD.rank == 0:
        if path_data.exists() and path_data.is_dir():
            print(f"Clearing data in {path_data}...")
            cleared_files = 0
            errors = 0
            # Iterate safely, only removing files directly within the target dir
            for item in path_data.iterdir():
                if item.is_file() and item.name.endswith((".gdf", ".dat")):
                    item.unlink()
                    cleared_files += 1
            print(
                f"Data clearing complete. Removed {cleared_files} files, encountered {errors} errors."
            )
        else:
            print(
                f"Data path {path_data} does not exist or is not a directory. Skipping clearing."
            )
    comm.barrier()  # Ensure all ranks wait until rank 0 is done


def load_input_data(traj_file="trajectory.txt", cmd_file="motor_commands.txt"):
    """Loads trajectory and motor command data."""
    try:
        trj = np.loadtxt(traj_file)
        motor_commands = np.loadtxt(cmd_file)

        if trj.shape[0] != motor_commands.shape[0]:
            raise ValueError(
                f"Trajectory ({traj_file}, length {trj.shape[0]}) and motor command ({cmd_file}, length {motor_commands.shape[0]}) files have different lengths."
            )
        # Handle 1D vs 2D data consistently
        if trj.ndim == 1:
            trj = trj.reshape(-1, 1)
        if motor_commands.ndim == 1:
            motor_commands = motor_commands.reshape(-1, 1)
        print(f"Loaded trajectory data with shape: {trj.shape}")
        print(f"Loaded motor command data with shape: {motor_commands.shape}")
        return trj, motor_commands
    except FileNotFoundError as e:
        print(
            f"Error loading input data: {e}. Ensure '{traj_file}' and '{cmd_file}' exist."
        )
        sys.exit(1)
    except ValueError as e:
        print(f"Error processing input data files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during input data loading: {e}")
        sys.exit(1)


# --- NEST Kernel Setup ---
def setup_nest_kernel(sim_params: dict, seed: int, path_data: str):
    """Resets and configures the NEST kernel."""

    nest.SetKernelStatus(
        {
            "resolution": sim_params["res"],
            "overwrite_files": True,  # Important for multiple trials/runs
            "data_path": path_data,
            # "print_time": True, # Optional: Print simulation progress
        }
    )
    print(
        f"NEST Kernel: Resolution: {sim_params['res']}ms, Seed: {seed}, Data path: {path_data}"
    )


# --- MUSIC Setup ---
def setup_music_interface(n_dof: int, N: int, msc_params: dict, spine_params: dict):
    """Creates MUSIC proxies for input and output."""
    n_total_neurons = 2 * N * n_dof

    out_port_name = msc_params["out_port"]
    in_port_name = msc_params["in_port"]
    latency_const = msc_params["const"]

    # Output proxy
    print(f"Creating MUSIC out proxy: port '{out_port_name}'")
    proxy_out = nest.Create(
        "music_event_out_proxy", 1, params={"port_name": out_port_name}
    )
    print(f"Created MUSIC out proxy: port '{out_port_name}'")

    # Input proxy
    proxy_in = nest.Create(
        "music_event_in_proxy", n_total_neurons, params={"port_name": in_port_name}
    )
    for i, n in enumerate(proxy_in):
        nest.SetStatus(n, {"music_channel": i})
    print(
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
    print(f"Set MUSIC acceptable latency for '{in_port_name}' to {latency} ms")

    return proxy_in, proxy_out


def connect_controller_to_music(
    controller: SingleDOFController, proxy_in, proxy_out, dof_id, N
):
    """Connects a single controller's inputs/outputs to MUSIC proxies."""
    print(f"connecting MUSIC inputs, with dof_id={dof_id} and N={N}")
    # Get PopView objects
    bs_p_view, bs_n_view = controller.get_brainstem_output_popviews()
    sn_p_view, sn_n_view = controller.get_sensory_input_popviews()
    # Connect Brainstem outputs (access .pop from PopView)
    if bs_p_view and bs_n_view:
        start_channel_out = 2 * N * dof_id
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
    path_data: str,
    controllers: list[SingleDOFController],
):
    """Runs the NEST simulation for the specified number of trials."""
    total_sim_time_ms = sim_params.get("timeMax", 1000.0) + sim_params.get(
        "timeWait", 0.0
    )  # Defaults if missing
    comm = MPI.COMM_WORLD

    # Define the base names of populations expected (match ControllerPopulations)
    all_pop_names = [
        "planner_p",
        "planner_n",
        "mc_ffwd_p",
        "mc_ffwd_n",
        "mc_fbk_p",
        "mc_fbk_n",
        "mc_out_p",
        "mc_out_n",
        "brainstem_p",
        "brainstem_n",
        "sn_p",
        "sn_n",
        "pred_p",
        "pred_n",
        "state_p",
        "state_n",
        "fbk_smooth_p",
        "fbk_smooth_n",
    ]
    # Filter names based on existing populations in the first controller
    if controllers:
        valid_pop_names = [
            name
            for name in all_pop_names
            if getattr(controllers[0].pops, name, None) is not None
        ]
        if comm.rank == 0:
            print(f"Populations found for analysis/collapse: {valid_pop_names}")
    else:
        valid_pop_names = []
        if comm.rank == 0:
            print("No controllers found, skipping simulation loop.")
        return  # Exit if no controllers

    for trial in range(n_trials):
        current_sim_start_time = nest.GetKernelStatus(
            "biological_time"
        )  # Get time before sim
        if comm.rank == 0:
            print(
                f"--- Simulating Trial {trial + 1}/{n_trials} ({total_sim_time_ms} ms) ---"
            )
            print(f"Current simulation time: {current_sim_start_time} ms")
        start_trial_time = timer()

        nest.Simulate(total_sim_time_ms)

        end_trial_time = timer()
        if comm.rank == 0:
            print(
                f"Trial {trial + 1} simulation finished at {nest.GetKernelStatus('biological_time')} ms. Wall clock time: {timedelta(seconds=end_trial_time - start_trial_time)}"
            )

        print("getting spike events from feedback smooth...")
        senders_refactored, times_refactored = controller.pops.fbk_smooth_p.get_events()
        print(f"--- Refactored fbk_smooth_p spikes ---")
        print(f"Num spikes: {len(times_refactored)}")
        if len(times_refactored) > 0:
            print(
                f"First spike time: {times_refactored[0]}, sender: {senders_refactored[0]}"
            )
            print(
                f"Last spike time: {times_refactored[-1]}, sender: {senders_refactored[-1]}"
            )

        # --- Data Collapsing ---
        # Assumes PopView was initialized with to_file=True and collapse_files works
        if comm.rank == 0:
            print("Attempting data collapsing...")
        start_collapse_time = timer()
        pops_for_collapse = []
        for name in valid_pop_names:
            dof_pop_views = [
                getattr(c.pops, name)
                for c in controllers
                if getattr(c.pops, name, None) is not None
            ]
            if dof_pop_views:
                pops_for_collapse.append(dof_pop_views)

        if pops_for_collapse:
            collapse_files(
                path_data + "/",  # TODO use paths not strings
                valid_pop_names,
                pops_for_collapse,
                len(controllers),
            )
            if comm.rank == 0:
                print("Data collapsing function executed.")
        else:
            if comm.rank == 0:
                print("No valid populations found to collapse.")

        end_collapse_time = timer()
        if comm.rank == 0:
            print(
                f"Trial {trial + 1} data collapsing took: {timedelta(seconds=end_collapse_time - start_collapse_time)}"
            )
            # --- Data Gathering (Optional - if collapse_files doesn't populate PopView) ---
            # print("Attempting to gather collapsed data into PopView objects...")
            # for name_idx, name in enumerate(valid_pop_names):
            #     try:
            #         # Construct expected filename based on PopView label convention
            #         # Assumes label passed to PopView was like 'dofX_popname'
            #         # and collapse_files creates 'dofX_popname.gdf'
            #         # This needs verification based on actual collapse_files behavior!
            #         for dof_idx, pop_view in enumerate(pops_for_collapse[name_idx]):
            #             base_label = pop_view.label # Get label used at creation
            #             if base_label: # Check if label exists (was to_file=True?)
            #                  collapsed_file_path = Path(path_data) / f"{base_label}.gdf"
            #                  if collapsed_file_path.exists():
            #                      data = np.loadtxt(collapsed_file_path)
            #                      if data.ndim == 2 and data.shape[1] == 2: # Check format
            #                          # Filter based on GIDs associated with this specific PopView
            #                          min_gid = pop_view.pop[0].global_id
            #                          max_gid = pop_view.pop[-1].global_id
            #                          mask = (data[:,0] >= min_gid) & (data[:,0] <= max_gid)
            #                          pop_view.gather_data(data[mask, 0], data[mask, 1])
            #                          # print(f"Gathered {np.sum(mask)} spikes for {base_label}")
            #                      else:
            #                          print(f"Warning: Data format incorrect or empty in {collapsed_file_path}")
            #                  else:
            #                      print(f"Warning: Collapsed file {collapsed_file_path} not found for gathering.")
            #     except Exception as e:
            #         print(f"Error gathering data for population type {name}: {e}")

    if comm.rank == 0:
        print("--- Simulation Finished ---")


# --- Plotting ---
# Move plotting functions to a separate file (e.g., plot_utils.py) recommended
# Example structure for a plotting function (adapt as needed):
def plot_controller_outputs(
    controllers: list[SingleDOFController],
    total_time_vect_concat: np.ndarray,
    path_fig_str: str,
    save_fig: bool = True,
):
    """Plots outputs for various populations across all controllers."""
    if MPI.COMM_WORLD.rank != 0:
        return  # Only rank 0 plots
    if not controllers:
        print("No controllers provided, skipping plotting.")
        return
    path_fig = Path(path_fig_str)
    njt = len(controllers)
    lgd = [f"DoF {i}" for i in range(njt)]

    print("Generating plots...")

    for i, controller in enumerate(controllers):
        print(f"Plotting for DoF {i}...")

        # Plot Planner
        fig, ax = plotPopulation(
            total_time_vect_concat,
            controller.pops.planner_p,
            controller.pops.planner_n,
            [],
            [],
            [],
            [],  # Empty lists for reference, time_vecs, legend, styles
            title=f"Planner {lgd[i]}",
            buffer_size=15,
            filepath=path_fig / f"planner_{i}.png",
        )

        # Plot Brainstem
        fig, ax = plotPopulation(
            total_time_vect_concat,
            controller.pops.brainstem_p,
            controller.pops.brainstem_n,
            [],
            [],
            [],
            [],  # Empty lists for reference, time_vecs, legend, styles
            title=f"Brainstem {lgd[i]}",
            buffer_size=15,
            filepath=path_fig / f"brainstem_{i}.png",
        )

        # Plot MC Output
        fig, ax = plotPopulation(
            total_time_vect_concat,
            controller.pops.mc_out_p,
            controller.pops.mc_out_n,
            [],
            [],
            [],
            [],  # Empty lists for reference, time_vecs, legend, styles
            title=f"MC Out {lgd[i]}",
            buffer_size=15,
            filepath=path_fig / f"mc_out_{i}.png",
        )

        # Plot State Estimator
        fig, ax = plotPopulation(
            total_time_vect_concat,
            controller.pops.state_p,
            controller.pops.state_n,
            [],
            [],
            [],
            [],  # Empty lists for reference, time_vecs, legend, styles
            title=f"State Out {lgd[i]}",
            buffer_size=15,
            filepath=path_fig / f"state_{i}.png",
        )

        # Plot Sensory Neurons
        fig, ax = plotPopulation(
            total_time_vect_concat,
            controller.pops.sn_p,
            controller.pops.sn_n,
            [],
            [],
            [],
            [],  # Empty lists for reference, time_vecs, legend, styles
            title=f"Sensory Out {lgd[i]}",
            buffer_size=15,
            filepath=path_fig / f"sensoryneuron_{i}.png",
        )

    print("Plot generation finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    start_script_time = timer()

    nest.ResetKernel()
    # Set seed before setting other statuses that might depend on RNG
    nest.rng_seed = SEED
    random.seed(SEED)
    np.random.seed(SEED)

    world = MPI.COMM_WORLD
    # last process is for receiver_plant
    group = world.group.Excl([world.Get_size() - 1])

    comm = world.Create_group(group)
    rank = comm.rank
    size = comm.size
    if rank == 0:
        print(f"--- Starting main script on {size} MPI rank(s) ---")

    # --- Load Configuration ---
    params = load_config("new_params.json")
    sim = Simulation()
    exp = Experiment()

    # --- Extract Parameters or Set Defaults ---
    # Module, Pop, Connection params (directly from JSON)
    module_params = params.get("modules", {})
    pops_params = params.get("pops", {})
    conn_params = params.get("connections", {})
    fig_path_str = params.get(
        "path", "./figures/"
    )  # Use path from JSON, provide default

    # Simulation parameters (provide defaults if not in JSON)
    # TODO: Consider adding a "simulation" section to new_params.json

    sim_params = params.get(
        "simulation",
        {
            "res": sim.resolution,  # ms - Simulation resolution
            "timeMax": sim.timeMax,  # ms - Duration of one trial segment
            "timeWait": sim.timeWait,  # ms - Pause/wait time after trial segment
            "n_trials": sim.n_trials,  # Number of trials to run
            "clear_old_data": True,  # Option to clear previous data
        },
    )

    # Experiment parameters (provide defaults if not in JSON)
    # TODO: Consider adding an "experiment" section to new_params.json
    exp_params = params.get(
        "experiment",
        {
            "seed": SEED,  # Generate random seed if not provided
            "N": 50,  # Default number of neurons per sub-population
            # Add other experiment-level info if needed
        },
    )
    seed = exp_params["seed"]
    N = exp_params["N"]

    # MUSIC configuration (provide defaults if not in JSON)
    # TODO: Consider adding a "music" section to new_params.json
    music_cfg = params.get(
        "music",
        {
            "out_port": "mot_cmd_out",
            "in_port": "fbk_in",
            "const": 1e-6,  # Latency constant
        },
    )

    # --- Environment and Path Setup ---
    setup_environment()  # Setup LD_LIBRARY_PATH, install modules
    path_data, path_fig = setup_paths(fig_path_str)
    if sim_params.get("clear_old_data", False):
        clear_old_data(path_data)

    # --- Load Input Data ---
    trj, motor_commands = load_input_data()  # Use default filenames
    njt = trj.shape[1]  # Infer number of DoFs
    if rank == 0:
        print(f"Inferred {njt} DoF(s) from input data.")

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

    # Verify data length against single trial time vector
    if len(trj) != len(single_trial_time_vect):
        print(
            f"Warning: Input data length ({len(trj)}) does not match single trial time vector length ({len(single_trial_time_vect)} based on timeMax={sim_params['timeMax']}, timeWait={sim_params['timeWait']}, res={res})."
        )
        # Option 1: Truncate data (might lose information)
        # if len(trj) > len(single_trial_time_vect):
        #     trj = trj[:len(single_trial_time_vect)]
        #     motor_commands = motor_commands[:len(single_trial_time_vect)]
        #     print("Truncated input data to match single trial duration.")
        # Option 2: Adjust simulation times (might not be desired)
        # Option 3: Assume data spans *all* trials (requires different handling in Controller) - Less likely based on original code
        print(
            "Proceeding with potentially mismatched data/time vector length. Check parameters."
        )

    # --- Network Construction ---
    start_network_time = timer()
    setup_nest_kernel(sim_params, seed, path_data)  # Reset kernel before building

    controllers = []
    if rank == 0:
        print(f"--- Constructing Network ({njt} DoF(s), N={N}) ---")
    for j in range(njt):
        if rank == 0:
            print(f"Creating controller for DoF {j}...")
        # Safely get module params using .get with empty dict as default
        mc_p = module_params.get("motor_cortex", {})
        plan_p = module_params.get("planner", {})
        spine_p = module_params.get("spine", {})
        state_p = module_params.get("state", {})
        state_se_p = module_params.get("state_se", {})  # Check if used by Controller

        controller = SingleDOFController(
            dof_id=j,
            N=N,
            total_time_vect=single_trial_time_vect,  # Pass single trial vector
            trajectory_slice=trj[:, j],
            motor_cmd_slice=motor_commands[:, j],
            # Pass parameter dicts extracted from JSON
            mc_params=mc_p,
            plan_params=plan_p,
            spine_params=spine_p,
            state_params=state_p,
            state_se_params=state_se_p,  # Pass if needed
            pops_params=pops_params,  # Pass the whole pops dict
            conn_params=conn_params,  # Pass the whole connections dict
            sim_params=sim_params,  # Pass simulation params (res, etc.)
            path_data=path_data,  # Pass data path (PopView might use internally)
            label_prefix="",  # No prefix in this example
        )
        controllers.append(controller)

    # --- MUSIC Setup and Connection ---
    spine_params = module_params["spine"]  # Get spine params again for MUSIC setup
    proxy_in, proxy_out = setup_music_interface(njt, N, music_cfg, spine_params)
    for j, controller in enumerate(controllers):
        if rank == 0:
            print(f"Connecting controller {j} to MUSIC...")
        connect_controller_to_music(controller, proxy_in, proxy_out, j, N)

    # --- Inter-Controller Connections (if any) ---
    # Add code here if controllers need to be connected to each other

    end_network_time = timer()
    if rank == 0:
        print(f"--- Network Construction Finished ---")
        print(
            f"Network construction wall clock time: {timedelta(seconds=end_network_time - start_network_time)}"
        )

    # --- Simulation ---
    run_simulation(sim_params, n_trials, path_data, controllers)

    # --- Plotting (Rank 0 Only) ---
    if rank == 0:
        print("--- Generating Plots ---")
        start_plot_time = timer()
        plot_controller_outputs(
            controllers, total_time_vect_concat, path_fig, save_fig=True
        )
        end_plot_time = timer()
        print(
            f"Plotting wall clock time: {timedelta(seconds=end_plot_time - start_plot_time)}"
        )

    # --- Final Timing ---
    end_script_time = timer()
    if rank == 0:
        total_duration = timedelta(seconds=end_script_time - start_script_time)
        print(f"--- Script Finished ---")
        print(f"Total wall clock time: {total_duration}")
