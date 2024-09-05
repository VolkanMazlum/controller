"""Planner class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt
import nest
import trajectories as tj

class Planner:

    #### Constructor (plant value is just for testing)
    def __init__(self, n, numJoints, time_vect, trj, kPlan=0.5, pathData="./data/", base_rate = 0.0, kp = 1200.0):
        # Time vector
        self.time_vect = time_vect
        
        # Gain to update the planned target
        self.kPlan = kPlan

        # Number of variables
        self.numJoints = numJoints

        # Encoded trajectory
        self.trj_j = trj

        print('Trj planner: ', len(self.trj_j))

        # General parameters of neurons
        params = {
            "base_rate": base_rate,
            "kp": kp
            }
        print('planner base rate: ', params["base_rate"])
        # Initialize population arrays
        # Create populations
        for i in range(self.numJoints):
            # Positive population (joint i)
            self.pops_p = nest.Create("tracking_neuron_nestml", n=n, params={"kp": kp, "base_rate": base_rate, "pos": True, "traj": self.trj_j, "simulation_steps": len(self.trj_j)})

            # Negative population (joint i)
            self.pops_n = nest.Create("tracking_neuron_nestml", n=n, params={"kp": kp, "base_rate": base_rate, "pos": False, "traj": self.trj_j, "simulation_steps": len(self.trj_j)})
