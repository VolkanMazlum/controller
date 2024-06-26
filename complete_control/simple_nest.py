import sys
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Adjust env vars to be able to import the NESTML-generated module
ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
new_path = ld_lib_path + ":"+"../nestml/target"
os.environ['LD_LIBRARY_PATH'] = new_path

import nest
sys.path.insert(1, '../')

from settings import Experiment, Simulation, Brain
from robot1j import Robot1J 
import mpi4py
from mpi4py import MPI
import random
import json
import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)


#### INITIALIZATION ####
# Load parameters from json file (could be also defined here directly)
f = open('new_params.json')
params = json.load(f)
f.close()

mc_params = params["modules"]["motor_cortex"]
plan_params = params["modules"]["planner"]
pops_params = params["pops"]
conn_params = params["connections"]

print(mc_params)

### Simulation
res = 0.1 #[ms]
time_span = 500.0 #[ms]
n_trial = 1
time_vect  = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

nest.ResetKernel()
nest.SetKernelStatus({"resolution": res})

### Experiment
# Initial and target position (end-effector space)
init_pos_ee = np.ndarray([1,2])
tgt_pos_ee = np.ndarray([1,2])

init_pos_ee[:] = [0.31,0.0]
tgt_pos_ee[:]  =[0.0,0.31]

# Dynamical system
robot_spec = {"mass": np.array([1.89]),
                    "links":np.array([0.31]),
                    "I": np.array([0.00189])}
dynSys     = Robot1J(robot=robot_spec)
dynSys.pos = dynSys.inverseKin(pos_external = init_pos_ee)
dynSys.vel = np.array([0.0])

njt = dynSys.numVariables()

# Initial and target position (joint space = the variable to be controlled is the elbow angle)
init_pos = dynSys.inverseKin( init_pos_ee )
tgt_pos  = dynSys.inverseKin( tgt_pos_ee )

# Compute minimum jerk trajectory

def minimumJerk(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =   6*(x_des-x_init)/np.power(T_max,5)
    b = -15*(x_des-x_init)/np.power(T_max,4)
    c =  10*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)
    e =  np.zeros(x_init.shape)
    g =  x_init

    pol = np.array([a,b,c,d,e,g])
    pp  = a*np.power(tmspn,5) + b*np.power(tmspn,4) + c*np.power(tmspn,3) + g

    return pp, pol
    
trj, pol = minimumJerk(init_pos[0], tgt_pos[0], time_vect) # Joint space (angle)
trj_ee = dynSys.forwardKin( trj ) # End-effector space


### Brain
N = 1 # Number of neurons for each (sub-)population
nTot = 2*N*njt # Total number of neurons (one positive and one negative population for each DOF)

### Install NESTML-generated module
nest.Install("controller_module")

### PLANNER: create and initialize the Planner population
# Input: target --> minimum jerk trajectory
# Output: spikes (firing rate proportional to elbow angle)
planner_p = nest.Create("tracking_neuron_nestml", n=N, params={"kp": plan_params["kp"], "base_rate": plan_params["base_rate"], "pos": True, "traj": trj.flatten().tolist(), "simulation_steps": len(trj.flatten().tolist())})

planner_n = nest.Create("tracking_neuron_nestml", n=N, params={"kp": plan_params["kp"], "base_rate": plan_params["base_rate"], "pos": False, "traj": trj.flatten().tolist(), "simulation_steps": len(trj.flatten().tolist())})


### FEEDFORWARD MOTOR CORTEX: create and initialize the Motor cortex populationù
# Input: target --> double derivative + inverse dynamics --> torque to be applied to elbow joint (motor command)
# Output: spikes (firing rate proportional to torque)

# Double derivative of the trajectory
def minimumJerk_ddt(x_init, x_des, timespan):
    T_max = timespan[ len(timespan)-1 ]
    tmspn = timespan.reshape(timespan.size,1)

    a =  120*(x_des-x_init)/np.power(T_max,5)
    b = -180*(x_des-x_init)/np.power(T_max,4)
    c =  60*(x_des-x_init)/np.power(T_max,3)
    d =  np.zeros(x_init.shape)

    pol = np.array([a,b,c,d])
    pp  = a*np.power(tmspn,3) + b*np.power(tmspn,2) + c*np.power(tmspn,1) + d
    ('pp: ', len(pp))
    return pp, pol
    
# Time and value of the minimum jerk curve
def minJerk_ddt_minmax(x_init, x_des, timespan):

    T_max   = timespan[ len(timespan)-1 ]
    t1      = T_max/2 - T_max/720 * np.sqrt(43200)
    t2      = T_max/2 + T_max/720 * np.sqrt(43200)
    pp, pol = minimumJerk_ddt(x_init, x_des, timespan)

    ext    = np.empty(shape=(2,x_init.size))
    ext[:] = 0.0
    t      = np.empty(shape=(2,x_init.size))
    t[:]   = 0.0

    for i in range(x_init.size):
        if (x_init[i]!=x_des[i]):
            tmp      = np.polyval( pol[:,i],[t1,t2] )
            ext[:,i] = np.reshape( tmp,(1,2) )
            t[:,i]   = np.reshape( [t1,t2],(1,2) )
    print('t: ', len(t))
    return t, ext
    
# Compute the torques via inverse dynamics
def generateMotorCommands(init_pos, des_pos, time_vector):
    # Last simulation time
    T_max = time_vector[ len(time_vector)-1 ]
    print(T_max)
    # Time and value of the minimum jerk curve
    ext_t, ext_val = minJerk_ddt_minmax(init_pos, des_pos, time_vector)

    # Approximate with sin function
    tmp_ext = np.reshape( ext_val[0,:], (1,njt) ) # First extreme (positive)
    tmp_sin = np.sin( (2*np.pi*time_vector/T_max) )
    mp_sin = np.reshape( tmp_sin,(tmp_sin.size,1) )

    # Motor commands: Inverse dynamics given approximated acceleration
    dt   = (time_vector[1]-time_vector[0])/1e3
    pos,pol  = minimumJerk(init_pos, des_pos, time_vector)
    vel  = np.gradient(pos,dt,axis=0)
    acc  = tmp_ext*tmp_sin
    mcmd = dynSys.inverseDyn(pos, vel, acc)
    return mcmd[0]

#motorCommands = generateMotorCommands(init_pos[0], tgt_pos[0], time_vect/1e-3)
motorCommands = generateMotorCommands(init_pos[0], tgt_pos[0], time_vect)

print(len(motorCommands.flatten().tolist()))
motor_p = nest.Create("tracking_neuron_nestml", n=N, params={"kp":mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': True, 'traj': motorCommands.flatten().tolist(), 'simulation_steps': len(motorCommands.flatten().tolist())})

motor_n = nest.Create("tracking_neuron_nestml", n=N, params={'kp':mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': False, 'traj': motorCommands.flatten().tolist(), 'simulation_steps': len(motorCommands.flatten().tolist())})

print('ok')
















