import sys
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import nest
sys.path.insert(1, '../')

from robot1j import Robot1J 
import json
import ctypes

# Time variables
res = 0.1 #[ms]
time_span = 500.0 #[ms]
n_trial = 1
time_vect  = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

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

## Compute minimum jerk trajectory (input to Planner)

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

## Save trajectory to file
np.savetxt('trajectory.txt', trj.flatten().tolist())

## Compute motor commands (input to Motor Cortex)
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
np.savetxt('motor_commands.txt', motorCommands.flatten().tolist())

'''
# Plot (test)
fig, ax = plt.subplots(2,1)
ax[0].plot(time_vect, trj)
ax[1].plot(time_vect, motorCommands)
plt.savefig('test.png')
'''
