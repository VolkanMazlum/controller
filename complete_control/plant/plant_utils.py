from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import structlog
from plant.plant_config import PlantConfig

_log = structlog.get_logger(__name__)


# --- Data Processing Functions ---
@dataclass
class DataArrays:
    pos_j_rad: np.ndarray
    vel_j_rad_s: np.ndarray
    pos_ee_m: np.ndarray
    vel_ee_m_s: np.ndarray
    spk_rate_pos_hz: np.ndarray
    spk_rate_neg_hz: np.ndarray
    spk_rate_net_hz: np.ndarray
    input_cmd_torque: np.ndarray
    input_cmd_total_torque: np.ndarray

    def __init__(self, num_total_steps, NJT):
        self.pos_j_rad = np.zeros([num_total_steps, NJT])
        self.vel_j_rad_s = np.zeros([num_total_steps, NJT])
        self.pos_ee_m = np.zeros([num_total_steps, 3])
        self.vel_ee_m_s = np.zeros([num_total_steps, 2])
        self.spk_rate_pos_hz = np.zeros([num_total_steps, NJT])
        self.spk_rate_neg_hz = np.zeros([num_total_steps, NJT])
        self.spk_rate_net_hz = np.zeros([num_total_steps, NJT])
        self.input_cmd_torque = np.zeros([num_total_steps, NJT])
        self.input_cmd_total_torque = np.zeros([num_total_steps, NJT])

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def compute_spike_rate(
    spikes: List[Tuple[float, int]],  # List of (timestamp, channel_id)
    weight: float,
    n_neurons: int,
    time_start: float,
    time_end: float,
) -> Tuple[float, int]:
    """
    Computes spike rate within a given time window.

    Args:
        spikes: List of spikes, where each spike is (timestamp, channel_id).
        weight: Weight associated with spikes (from original code, seems to be w).
        n_neurons: Number of neurons in the population.
        time_start: Start of the time buffer for rate calculation.
        time_end: End of the time buffer for rate calculation.

    Returns:
        A tuple (spike_rate_hz, weighted_spike_count).
    """
    if not spikes:
        return 0.0, 0

    # Filter spikes within the time window [time_start, time_end)
    # Using a more direct iteration for potentially better performance with many spikes
    # than converting to numpy array first for just filtering.
    count = 0
    for t, _ in spikes:
        if time_start <= t < time_end:
            count += 1

    weighted_count = weight * count
    duration = time_end - time_start

    if duration <= 0 or n_neurons <= 0:
        # Avoid division by zero or meaningless rates
        _log.warning(
            "Invalid duration or n_neurons for rate calculation",
            duration=duration,
            n_neurons=n_neurons,
        )
        return 0.0, int(weighted_count)

    rate_hz = weighted_count / (duration * n_neurons)
    return rate_hz, int(weighted_count)


def record_step_data(
    data_arrays: DataArrays,
    step: int,
    joint_pos_rad: float,
    joint_vel_rad_s: float,
    ee_pos_m: List[float],  # [x,y,z]
    ee_vel_m_s: List[float],  # [vx, vz] as per original slicing, or [vx,vy,vz]
    spk_rate_pos_hz: float,
    spk_rate_neg_hz: float,
    spk_rate_net_hz: float,
    input_cmd_torque: float,
    # Add other data points as needed, e.g., total command if perturbations exist
) -> None:
    """
    Records data for the current simulation step into the data_arrays.
    Assumes single joint (NJT=1).
    """
    if step < 0 or step >= data_arrays.pos_j_rad.shape[0]:
        _log.error(
            "Step index out of bounds for data_arrays",
            step=step,
            max_steps=data_arrays["pos_j_rad"].shape[0],
        )
        # Let it raise IndexError if out of bounds, as per user feedback.
        # No, this is a logic error, not a malformed input. We should handle it.
        return

    data_arrays["pos_j_rad"][step, 0] = joint_pos_rad
    data_arrays["vel_j_rad_s"][step, 0] = joint_vel_rad_s
    data_arrays["pos_ee_m"][step, :] = ee_pos_m[0:3]  # Ensure it's [x,y,z]

    # Original code saved ee_vel as [vx, vz]. If ee_vel_m_s is [vx,vy,vz], slice it.
    if len(ee_vel_m_s) == 3:
        data_arrays["vel_ee_m_s"][step, :] = [ee_vel_m_s[0], ee_vel_m_s[2]]
    elif len(ee_vel_m_s) == 2:  # Assumes it's already [vx, vz]
        data_arrays["vel_ee_m_s"][step, :] = ee_vel_m_s
    else:
        _log.error("Unexpected ee_vel_m_s length", length=len(ee_vel_m_s))
        raise ValueError("Unexpected ee_vel_m_s length")

    data_arrays["spk_rate_pos_hz"][step, 0] = spk_rate_pos_hz
    data_arrays["spk_rate_neg_hz"][step, 0] = spk_rate_neg_hz
    data_arrays["spk_rate_net_hz"][step, 0] = spk_rate_net_hz
    data_arrays["input_cmd_torque"][step, 0] = input_cmd_torque
    # Assuming input_cmd_total_torque is same as input_cmd_torque if no perturbations
    data_arrays["input_cmd_total_torque"][step, 0] = input_cmd_torque


def _save_spikes_to_file(filepath: Path, spikes: List[Tuple[float, int]]):
    """Helper to save spike data, handling empty lists."""
    if spikes:
        # Sort by timestamp before saving
        spikes.sort(key=lambda x: x[0])
        np.savetxt(filepath, np.array(spikes), fmt="%3.4f\t%d", delimiter="\t")
    else:
        # Create an empty file if no spikes, as per original behavior
        filepath.touch()
        # np.savetxt(filepath, np.array([])) # This would also work


def save_all_data(
    config: PlantConfig,
    data_arrays: DataArrays,
    received_spikes: Dict[
        str, List[List[Tuple[float, int]]]
    ],  # {'pos': [spikes_j0, ...], 'neg': [spikes_j0, ...]}
    sensory_spikes: Dict[
        str, List[List[Tuple[float, int]]]
    ],  # {'p': [spikes_j0, ...], 'n': [spikes_j0, ...]}
) -> None:
    """
    Saves all collected simulation data to files.
    """
    pth_dat_bullet = config.run_paths.data_bullet
    _log.info(f"Saving all simulation data at {pth_dat_bullet}")

    # Save spike rates
    # TODO make all these constants
    np.savetxt(
        pth_dat_bullet / "motNeur_rate_pos.csv",
        data_arrays["spk_rate_pos_hz"],
        delimiter=",",
    )
    np.savetxt(
        pth_dat_bullet / "motNeur_rate_neg.csv",
        data_arrays["spk_rate_neg_hz"],
        delimiter=",",
    )
    np.savetxt(
        pth_dat_bullet / "motNeur_rate_net.csv",
        data_arrays["spk_rate_net_hz"],
        delimiter=",",
    )

    # Save positions and velocities
    np.savetxt(
        pth_dat_bullet / "pos_real_joint.csv", data_arrays.pos_j_rad, delimiter=","
    )
    np.savetxt(
        pth_dat_bullet / "vel_real_joint.csv", data_arrays.vel_j_rad_s, delimiter=","
    )
    np.savetxt(pth_dat_bullet / "pos_real_ee.csv", data_arrays.pos_ee_m, delimiter=",")
    np.savetxt(
        pth_dat_bullet / "vel_real_ee.csv", data_arrays.vel_ee_m_s, delimiter=","
    )

    # To get full desired EE trajectory, we'd need to apply forward kinematics to the tiled joint trajectory
    # desired_ee_trj = np.array([config.DYN_SYS.forwardKin(q_joint) for q_joint in tiled_desired_joint_trj])
    # np.savetxt(pth_dat_bullet / "pos_des_ee.csv", desired_ee_trj, delimiter=",") # Placeholder

    # Save motor commands
    # np.savetxt(pth_dat_bullet / "inputCmd_des.csv", inputDes, delimiter=",") # 'inputDes' needs to be calculated
    np.savetxt(
        pth_dat_bullet / "inputCmd_motNeur.csv",
        data_arrays.input_cmd_torque,
        delimiter=",",
    )
    np.savetxt(
        pth_dat_bullet / "inputCmd_tot.csv",
        data_arrays.input_cmd_total_torque,
        delimiter=",",
    )

    # Save received spikes (motor commands from MUSIC)
    for j in range(config.NJT):
        _save_spikes_to_file(
            pth_dat_bullet / f"motNeur_inSpikes_j{j}_p.txt", received_spikes["pos"][j]
        )
        _save_spikes_to_file(
            pth_dat_bullet / f"motNeur_inSpikes_j{j}_n.txt", received_spikes["neg"][j]
        )

    # Save sensory spikes (output to MUSIC)
    for j in range(config.NJT):
        _save_spikes_to_file(
            pth_dat_bullet / f"sensNeur_outSpikes_j{j}_p.txt", sensory_spikes["p"][j]
        )
        _save_spikes_to_file(
            pth_dat_bullet / f"sensNeur_outSpikes_j{j}_n.txt", sensory_spikes["n"][j]
        )
    _log.info("Finished saving all data.")


# --- Plotting Functions ---


def plot_joint_space(
    config: PlantConfig,
    time_vector_s: np.ndarray,
    pos_j_rad_actual: np.ndarray,
    desired_trj_joint_rad: np.ndarray,
    save_fig: bool = True,
) -> None:
    """Plots joint space position (actual vs desired)."""
    pth_fig_receiver = config.run_paths.figures_receiver
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure()
    plt.plot(
        time_vector_s, pos_j_rad_actual[:, 0], linewidth=2, label="Actual Joint Angle"
    )

    # Construct full desired trajectory for plotting
    # The config.trajectory_joint_single_trial_rad is for one trial.
    # We need to tile it or use a pre-computed full desired trajectory.
    # For now, let's plot only actual if full desired isn't readily available.
    # Or, plot the single trial desired trajectory overlaid repeatedly.

    # Example: Overlaying the single trial desired trajectory for each trial period
    single_trial_steps = len(config.time_vector_single_trial_s)
    desired_single_trial = desired_trj_joint_rad

    full_desired_plot = np.full_like(pos_j_rad_actual[:, 0], np.nan)
    for trial_n in range(config.N_TRIALS):
        start_idx = trial_n * single_trial_steps
        end_idx = start_idx + len(desired_single_trial)
        if end_idx <= len(full_desired_plot):
            full_desired_plot[start_idx:end_idx] = desired_single_trial
        else:  # partial last trial
            len_to_copy = len(full_desired_plot) - start_idx
            if len_to_copy > 0:
                full_desired_plot[start_idx:] = desired_single_trial[:len_to_copy]

    plt.plot(
        time_vector_s,
        full_desired_plot,
        linestyle=":",
        linewidth=2,
        label="Desired Joint Angle (Per Trial)",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")
    plt.title("Joint Space Position")
    plt.legend()
    plt.ylim((0.0, 1.6))  # As in original
    if save_fig:
        filepath = pth_fig_receiver / f"position_joint_{timestamp}.png"
        plt.savefig(filepath)
        _log.info(f"Saved joint space plot at {filepath}")
    plt.close()


def plot_ee_space(
    config: PlantConfig,
    desired_start_ee: np.ndarray,
    desired_end_ee: np.ndarray,
    actual_traj_ee: np.ndarray,  # Shape (num_time_steps, 3) for x,y,z
    save_fig: bool = True,
) -> None:
    """Plots end-effector space trajectory."""
    pth_fig_receiver = config.run_paths.figures_receiver
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_init_marker_xz = [desired_start_ee[0], desired_start_ee[2]]
    plot_target_marker_xz = [desired_end_ee[0], desired_end_ee[2]]

    plt.figure()
    single_trial_steps = len(config.time_vector_single_trial_s)

    for trial_n in range(config.N_TRIALS):
        start_idx = trial_n * single_trial_steps
        end_idx = start_idx + single_trial_steps
        if end_idx > actual_traj_ee.shape[0]:
            end_idx = actual_traj_ee.shape[0]

        # Plot X (pos_ee_m_actual[:,0]) vs Z (pos_ee_m_actual[:,2])
        plt.plot(
            actual_traj_ee[start_idx:end_idx, 0],
            actual_traj_ee[start_idx:end_idx, 2],
            "k",
            label="Trajectory" if trial_n == 0 else None,
        )
        plt.plot(
            actual_traj_ee[end_idx - 1, 0],
            actual_traj_ee[end_idx - 1, 2],
            marker="x",
            color="k",
            label="Reached (End of Trial)" if trial_n == 0 else None,
        )

    plt.plot(
        plot_init_marker_xz[0],
        plot_init_marker_xz[1],
        marker="o",
        color="blue",
        label="Config Start EE",
    )
    plt.plot(
        plot_target_marker_xz[0],
        plot_target_marker_xz[1],
        marker="o",
        color="red",
        label="Config Target EE",
    )

    plt.axis("equal")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Z (m)")
    plt.title("End-Effector Trajectory")
    plt.legend()
    if save_fig:
        filepath = pth_fig_receiver / f"position_ee_{timestamp}.png"
        plt.savefig(filepath)
        _log.info(f"Saved end-effector space plot at {filepath}")
    plt.close()


def plot_motor_commands(
    config: PlantConfig,
    time_vector_s: np.ndarray,
    input_cmd_torque_actual: np.ndarray,
    # input_cmd_torque_desired: np.ndarray, # If available
    save_fig: bool = True,
) -> None:
    """Plots motor commands (actual vs desired if available)."""
    pth_fig_receiver = config.run_paths.figures_receiver
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond_str = "refactored"  # Placeholder for condition string from original plots

    plt.figure()
    plt.plot(time_vector_s, input_cmd_torque_actual[:, 0], label="Actual Motor Command")
    # if input_cmd_torque_desired is not None:
    #     plt.plot(time_vector_s, input_cmd_torque_desired[:,0], linestyle=':', label="Desired Motor Command")
    plt.xlabel("Time (s)")
    plt.ylabel("Motor Command (Torque N.m)")  # Assuming torque
    plt.title("Motor Commands")
    plt.legend()
    if save_fig:
        filepath = pth_fig_receiver / f"{cond_str}_motCmd_{timestamp}.png"
        plt.savefig(filepath)
        _log.info(f"Saved motor commands plot at {filepath}")
    plt.close()


def plot_errors_per_trial(
    config: PlantConfig,
    errors_list: List[float],  # List of final error per trial
    save_fig: bool = True,
) -> None:
    """Plots the final error for each trial."""
    if not errors_list:
        _log.info("No errors to plot.")
        return

    pth_fig_receiver = config.run_paths.figures_receiver
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond_str = "refactored"

    plt.figure()
    plt.plot(range(1, len(errors_list) + 1), errors_list, marker="o")
    plt.xlabel("Trial Number")
    plt.ylabel("Final Error (rad or m, depending on error metric)")
    plt.title("Error per Trial")
    plt.grid(True)
    if save_fig:
        filepath = pth_fig_receiver / f"{cond_str}_error_ee_trial_{timestamp}.png"
        plt.savefig(filepath)
        _log.info(f"Saved error per trial plot at at {filepath}")
    plt.close()
