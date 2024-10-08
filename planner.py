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
from population_view import PopView
from util import AddPause


class Planner:
    #### Constructor (plant value is just for testing)
    def __init__(self, n, numJoints, time_vect, trj, pathData, kPlan=0.5, base_rate = 0.0, kp = 1200.0):
        # Time vector
        self.time_vect = time_vect
        
        # Gain to update the planned target
        self.kPlan = kPlan

        # Number of variables
        self.numJoints = numJoints

        # Encoded trajectory
        self.trj_j = trj

        # General parameters of neurons
        params = {
            "base_rate": base_rate,
            "kp": kp
            }

        # Initialize population arrays
        self.pops_p = []
        self.pops_n = []

        # Create populations
        #print('base_rate: ', base_rate)
        for i in range(self.numJoints):
            # Positive population (joint i)
            #tmp_pop_p = nest.Create("tracking_neuron_planner_nestml", n=n, params={"kp": params["kp"], "base_rate": params["base_rate"], "pos": True, "traj": self.trj_j, "simulation_steps": len(self.trj_j)})
            tmp_pop_p = nest.Create("tracking_neuron_nestml", n=n)
            for idx, neuron in enumerate(tmp_pop_p):
                nest.SetStatus(neuron,{"kp": params["kp"], "base_rate": params["base_rate"], "pos": True, "traj": self.trj_j, "simulation_steps": len(self.trj_j)} )
            
            self.pops_p.append( PopView(tmp_pop_p, self.time_vect, to_file=True, label="planner_p") )

            # Negative population (joint i)
            tmp_pop_n = nest.Create("tracking_neuron_nestml", n=n, params={"kp": params["kp"], "base_rate": params["base_rate"], "pos": False, "traj": self.trj_j, "simulation_steps": len(self.trj_j)})
            self.pops_n.append( PopView(tmp_pop_n, self.time_vect, to_file=True, label="planner_n") )
