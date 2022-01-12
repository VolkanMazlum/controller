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
    def __init__(self, n, time_vect, target, plant, kPlan=1.0, pathData="./data/", pause_len=0.0, **kwargs):

        # Path where to save the data file
        self.pathData = pathData

        # Gain to update the planned target
        self.kPlan = kPlan

        # Model of the plant to be controlled
        self.plant = plant
        self.numJoints = plant.numVariables()
        self.init_endEff = plant.forwardKin( plant.pos )

        # Time vector
        self.time_vect = time_vect
        self.time_span = self.time_vect[ len(self.time_vect)-1 ]

        # Time pause after task execution
        self.pause_len = pause_len

        # Desired and planned target location, and target error in end-effector space
        self.target_des = target
        self.target_plan = target
        self.error_plan = 0.0

        # Desired and planned trajectory in end-effector space
        self.traj_des  = self.generateEndEffTraj(self.init_endEff, self.target_des)
        self.traj_plan = self.generateEndEffTraj(self.init_endEff, self.target_plan)

        # Planned trajectory in joint space
        self.traj_plan_j = self.generateJointTraj(self.traj_plan)

        # General parameters of neurons
        params = {
            "base_rate": 0.0,
            "kp": 1.0,
            }
        params.update(kwargs)

        # Initialize population arrays
        self.pops_p = []
        self.pops_n = []

        # Create populations
        for i in range(self.numJoints):
            # File where the trajectory is saved (from trajectory object)
            traj_file = "./data/joint" + str(i) + ".dat"

            # Positive population (joint i)
            tmp_pop_p = nest.Create("tracking_neuron", n=n, params=params)
            nest.SetStatus(tmp_pop_p, {"pos": True, "pattern_file": traj_file})
            self.pops_p.append( PopView(tmp_pop_p,self.time_vect) )

            # Negative population (joint i)
            tmp_pop_n = nest.Create("tracking_neuron", n=n, params=params)
            nest.SetStatus(tmp_pop_n, {"pos": False, "pattern_file": traj_file})
            self.pops_n.append( PopView(tmp_pop_n,self.time_vect) )


    #### Target (end-effector space)

    def getTargetDes(self):
        return self.target_des

    def setTargetDes(self, tgt):
        self.target_des = tgt

    def getTargetPlan(self):
        return self.target_plan

    def setTargetPlan(self, tgt):
        self.target_plan = tgt
        self.traj_plan   = self.generateEndEffTraj(self.init_endEff, self.target_plan)   # Update planned trajectory (end-effector space)
        self.traj_plan_j = self.generateJointTraj(self.traj_plan)

    def updateTarget(self, error):
        self.error_plan  = error                                                         # Record error
        self.target_plan = self.getTargetPlan()-self.kPlan*error                         # Update planned traget
        self.traj_plan   = self.generateEndEffTraj(self.init_endEff, self.target_plan)   # Update planned trajectory (end-effector space)
        self.traj_plan_j = self.generateJointTraj(self.traj_plan)                        # Update planned trajectory (joint space)


    #### Trajectory (end-effector space)

    def getTrajDes(self):
        return self.traj_des

    def getTrajPlan(self):
        return self.traj_plan

    def generateEndEffTraj(self,init,target):
        trj, pol = tj.minimumJerk(init, target, self.time_vect)
        return trj

    def setTrajDes(self):
        self.generateEndEffTraj(self.init_endEff,self.target_des)

    def setTrajPlan(self):
        self.generateEndEffTraj(self.init_endEff,self.target_plan)


    #### Trajectory (joint space)

    def getTrajPlan_j(self):
            return self.traj_plan_j

    def generateJointTraj(self, trj_endEff):
        trj_j = self.plant.inverseKin( trj_endEff )

        nj = trj_j.shape[1]
        if nj!=self.numJoints:
            raise Exception("Number of joint is different from number of columns")

        # Include pause into the trajectory
        res = nest.GetKernelStatus({"resolution"})[0]
        time_bins = trj_j.shape[0] + int(self.pause_len/res)
        trj_j_pause = np.zeros((time_bins, trj_j.shape[1])) 
        for i in range(nj):
            trj_j_pause[:,i] = AddPause(trj_j[:,i], self.pause_len, res)

        # save joint trajectories into files
        # NOTE: THIS OVERWRITES EXISTING TRAJECTORIES
        for i in range(nj):
            traj_file = self.pathData + "joint" + str(i) + ".dat"
            a_file = open(traj_file, "w")
            np.savetxt( a_file, trj_j_pause[:,i] )
            a_file.close()

        return trj_j_pause
