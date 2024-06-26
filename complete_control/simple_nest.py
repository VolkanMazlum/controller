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
#import mpi4py
#from mpi4py import MPI
import random
import json
import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)


############## INITIALIZATION #############
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

# Compute minimum jerk trajectory (input to Planner)

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

# Compute motor commands (input to Motor Cortex)
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

motorCommands = generateMotorCommands(init_pos[0], tgt_pos[0], time_vect/1e3)
#motorCommands = generateMotorCommands(init_pos[0], tgt_pos[0], time_vect)

fig, ax = plt.subplots(1,1)
plt.plot(time_vect, motorCommands)
plt.savefig('mcmd.png')

### Brain
N = 50 # Number of neurons for each (sub-)population
nTot = 2*N*njt # Total number of neurons (one positive and one negative population for each DOF)

####################### CREATE NODES/MODULES #####################
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
motor_p = nest.Create("tracking_neuron_nestml", n=N, params={"kp":mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': True, 'traj': motorCommands.flatten().tolist(), 'simulation_steps': len(motorCommands.flatten().tolist())})

motor_n = nest.Create("tracking_neuron_nestml", n=N, params={'kp':mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': False, 'traj': motorCommands.flatten().tolist(), 'simulation_steps': len(motorCommands.flatten().tolist())})

### BRAINSTEM
# Input: spikes from FFWD motor cortex
# Output: smoothed profile of firing rate over time
for j in range(njt):
    # Positive neurons
    brainstem_p = nest.Create ("basic_neuron_nestml", N)
    nest.SetStatus(brainstem_p, {"kp": pops_params["brain_stem"]["kp"], "pos": True, "buffer_size": pops_params["brain_stem"]["buffer_size"], "base_rate": pops_params["brain_stem"]["base_rate"], "simulation_steps": len(time_vect)})
    # Negative neurons
    brainstem_n = nest.Create ("basic_neuron_nestml", N)
    nest.SetStatus(brainstem_n, {"kp": pops_params["brain_stem"]["kp"], "pos": False, "buffer_size": pops_params["brain_stem"]["buffer_size"], "base_rate": pops_params["brain_stem"]["base_rate"], "simulation_steps": len(time_vect)})

### Connections between FFWD MC and brainstem
for j in range(njt):
    nest.Connect(motor_p, brainstem_p, "all_to_all", {"weight": conn_params["mc_out_brain_stem"]["weight"], "delay": conn_params["mc_out_brain_stem"]["delay"]})

    nest.Connect(motor_n ,brainstem_n, "all_to_all", {"weight": -conn_params["mc_out_brain_stem"]["weight"], "delay": conn_params["mc_out_brain_stem"]["delay"]})

### DEVICES
spikedetector_planner_pos = nest.Create("spike_recorder", params={"label": "Planner pos"})
spikedetector_planner_neg = nest.Create("spike_recorder", params={"label": "Planner neg"})

spikedetector_brain_stem_pos = nest.Create("spike_recorder", params={"label": "Brain stem pos"})
spikedetector_brain_stem_neg = nest.Create("spike_recorder", params={"label": "Brain stem neg"})

spikedetector_motor_cortex_pos = nest.Create("spike_recorder", params={"label": "Motor cortex pos"})
spikedetector_motor_cortex_neg = nest.Create("spike_recorder", params={"label": "Motor cortex neg"})

nest.Connect(planner_p, spikedetector_planner_pos)
nest.Connect(planner_n, spikedetector_planner_neg)

nest.Connect(brainstem_p, spikedetector_brain_stem_pos)
nest.Connect(brainstem_n, spikedetector_brain_stem_neg)

nest.Connect(motor_p, spikedetector_motor_cortex_pos)
nest.Connect(motor_n, spikedetector_motor_cortex_neg)

########## SIMULATION #############
for trial in range(n_trial):
   nest.Simulate(time_span)

########## PLOTTING ############### (test, poi da rimuovere)
## Planner
plan_data_p = nest.GetStatus(spikedetector_planner_pos, keys= "events")[0]
plan_data_n = nest.GetStatus(spikedetector_planner_neg, keys= "events")[0]
ts_p = plan_data_p["times"]
ts_n = plan_data_n["times"]
y_p = plan_data_p["senders"]
y_p = [n_id - min(y_p) + 1 for n_id in y_p]
y_n = plan_data_n["senders"]
y_n = [-n_id + min(y_n) + 1 for n_id in y_n]
for i in range(njt):
   fig, ax = plt.subplots(2,1)
   ax[0].scatter(ts_p, y_p, marker='.', s=1,c="r")
   ax[0].scatter(ts_n, y_n, marker='.', s=1)
   ax[0].set_ylabel("raster")
   #brain_stem_new_p[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='r',label='brainstem')
   #brain_stem_new_n[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='b',label='brainstem')
   #ax[1].set_xlabel("time (ms)")
   #plt.suptitle("Positive")
plt.savefig("planner_pos.png")

## Motor cortex
mc_data_p = nest.GetStatus(spikedetector_motor_cortex_pos, keys= "events")[0]
mc_data_n = nest.GetStatus(spikedetector_motor_cortex_neg, keys= "events")[0]
ts_p = mc_data_p["times"]
ts_n = mc_data_n["times"]
y_p = mc_data_p["senders"]
y_p = [n_id - min(y_p) + 1 for n_id in y_p]
y_n = mc_data_n["senders"]
y_n = [-n_id + min(y_n) + 1 for n_id in y_n]
for i in range(njt):
   fig, ax = plt.subplots(2,1)
   ax[0].scatter(ts_p, y_p, marker='.', s=1,c="r")
   ax[0].scatter(ts_n, y_n, marker='.', s=1)
   ax[0].set_ylabel("raster")
   #brain_stem_new_p[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='r',label='brainstem')
   #brain_stem_new_n[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='b',label='brainstem')
   #ax[1].set_xlabel("time (ms)")
   #plt.suptitle("Positive")
plt.savefig("mc_pos.png")

## Brainstem
bs_data_p = nest.GetStatus(spikedetector_brain_stem_pos, keys= "events")[0]
bs_data_n = nest.GetStatus(spikedetector_brain_stem_neg, keys= "events")[0]
ts_p = bs_data_p["times"]
ts_n = bs_data_n["times"]
y_p = bs_data_p["senders"]
y_p = [n_id - min(y_p) + 1 for n_id in y_p]
y_n = bs_data_n["senders"]
y_n = [-n_id + min(y_n) + 1 for n_id in y_n]
for i in range(njt):
   fig, ax = plt.subplots(2,1)
   ax[0].scatter(ts_p, y_p, marker='.', s=1,c="r")
   ax[0].scatter(ts_n, y_n, marker='.', s=1)
   ax[0].set_ylabel("raster")
   #brain_stem_new_p[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='r',label='brainstem')
   #brain_stem_new_n[i].plot_rate(time_vect_paused,ax=ax[1],bar=True,color='b',label='brainstem')
   #ax[1].set_xlabel("time (ms)")
   #plt.suptitle("Positive")
plt.savefig("brainstem_pos.png")


########## COMPUTE OUTPUT (net firing rate)









