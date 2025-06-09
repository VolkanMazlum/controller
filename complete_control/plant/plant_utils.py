from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import structlog
from config.plant_config import PlantConfig

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
