from pathlib import Path

import numpy as np
from Controller import Controller
from mpi4py import MPI

from population_view import plotPopulation


def plot_controller_outputs(
    controllers: list[Controller],
    total_time_vect_concat: np.ndarray,
    path_fig_str: str,
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
