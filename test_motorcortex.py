#!/usr/bin/env python3

import nest
import numpy as np
import matplotlib.pyplot as plt
import trajectories as tj
import time

from motorcortex import MotorCortex
from population_view import plotPopulation
from util import savePattern
from population_view import PopView
from pointMass import PointMass
from robot1j import Robot1J

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

# Randomize
msd = int( time.time() * 1000.0 )
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})

flagSaveFig = False
figPath = './fig/motorcortex/'
pthDat = "./data/"

# # Positions in end-effector space (meters)
# init_pos = np.array([0.0,0.0]) # Initial position
# tgt_pos  = np.array([0.5,0.5]) # Desired target
# tgt_real = np.array([0.5,0.2]) # Final position reached

# preciseControl = False

# # Dynamical system
# m          = 2.0
# ptMass     = PointMass(mass=m)
# ptMass.pos = ptMass.inverseKin(init_pos) # Place dyn sys in desired initial position
# ptMass.vel = np.array([0.0,0.0])         # with zero initial velocity
# njt        = ptMass.numVariables()



# End effector space
pos_i    = np.ndarray([1,2]) # Initial position
pos_i[:] = [2.0,0.0]
pos_tgt = np.ndarray([1,2])
pos_tgt[:] = [0.0, 2.0] # Desired target
#final  = np.array([0.25,0.43]) # Exemplary reached target (this will be the output of a simulation)
#kpl    = 1.0                 # Coefficient across-trial plan adjustement


# Dynamical system
robot_spec = {"mass": np.array([1.0]),
            "links":np.array([2.0])}
rob     = Robot1J(robot=robot_spec)
angles_i = rob.inverseKin(pos_external = pos_i) # Place dyn sys in desired initial position
rob.ang_vel = np.array([0.0])      # with zero initial velocity
njt        = rob.numVariables()
N = 50   # Number neurons
target = rob.inverseKin(pos_tgt)
final = 1.50

# Neuron neurons
N = 50

# Time span and time vector expressed in ms (as all the rest in NEST)
time_span = 1000.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

# Desired and real trajectories

trj,pol             = tj.minimumJerk(angles_i[0], target, time_vect)                   # joint space
trj_real ,pol        = tj.minimumJerk(angles_i[0], final, time_vect)              # joint space

trj_ee     = rob.forwardKin(trj)  # end-eff space
trj_ee_real = rob.forwardKin(trj_real) # end-eff space
# If one wants to zero the error
#trj_real = trj

# Error in joint trajectory
trj_err = trj-trj_real
#trj_err = np.zeros(shape=(time_vect.size,njt))
#trj_err[:,0] = -15.0
savePattern(trj_err, pthDat+"error")


####### Error population (goes into motor cortex feedback)
err_param = {
    "base_rate": 0.0,
    "kp": 10.0
    }

err_pop_p = []
err_pop_n = []
for i in range(njt):
    # Positive population (joint i)
    tmp_pop_p = nest.Create("tracking_neuron", n=N, params=err_param)
    nest.SetStatus(tmp_pop_p, {"pos": True, "pattern_file": pthDat+"error"+"_"+str(i)+".dat"})
    err_pop_p.append( PopView(tmp_pop_p,time_vect) )

    # Negative population (joint i)
    tmp_pop_n = nest.Create("tracking_neuron", n=N, params=err_param)
    nest.SetStatus(tmp_pop_n, {"pos": False, "pattern_file": pthDat+"error"+"_"+str(i)+".dat"})
    err_pop_n.append( PopView(tmp_pop_n,time_vect) )


####### Motor cortex
mc_param = {
    "ffwd_base_rate": 0.0, # Feedforward neurons
    "ffwd_kp": 10.0,
    "fbk_base_rate": 0.0,  # Feedback neurons
    "fbk_kp": 10.0,
    "out_base_rate": 0.0,  # Summation neurons
    "out_kp":1.0,
    "wgt_ffwd_out": 1.0,   # Connection weight from ffwd to output neurons (must be positive)
    "wgt_fbk_out": 1.0,    # Connection weight from fbk to output neurons (must be positive)
    "buf_sz": 50.0         # Size of the buffer to compute spike rate in ffwd and out populations
    }
preciseControl =True
mc = MotorCortex(N, time_vect, trj, rob, pthDat, preciseControl, **mc_param)


# ####### Connections (error to motor cortex feedback)
for i in range(njt):
    err_pop_p[i].connect(mc.fbk_p[i], rule='one_to_one', w= 1.0)
    err_pop_n[i].connect(mc.fbk_n[i], rule='one_to_one', w=-1.0)


############### TEST ANALOG SIGNALS

lgd = ['x plan','y plan','x real','y real']

# Visualize initial planned trajectory
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot( time_vect,mc.getJointTrajectory() )
ax[0].plot( time_vect,trj_real,linestyle=':')
ax[0].set_ylabel("Motor plan (m)")
ax[0].legend(lgd)
ax[1].plot( time_vect,mc.getMotorCommands() )
ax[1].set_ylabel("Force (N)")
ax[1].set_xlabel("Time (ms)")

if flagSaveFig:
    plt.savefig(figPath+"motctx_trj_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".png",format="png")
    plt.savefig(figPath+"motctx_trj_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".svg",format="svg")


########## TEST CONTROL

# mcmd = mc.getMotorCommands()
# dt   = (time_vect[1]-time_vect[0])/1e3
# s    = np.zeros(shape=mcmd.shape)
#
# for i in range(time_vect.size):
#     s[i,:] = ptMass.pos
#     ptMass.integrateTimeStep(mcmd[i,:], dt)
#
# plt.figure()
# plt.plot(s)
# plt.plot(trj)
# plt.show()


###################### SIMULATE

# Simulate
nest.Simulate(time_span)


###################### PLOTTING

lgd = ['x','y']

axv = []
fgv = []
max_y = np.empty(shape=(njt*3,2))

# Populations
for i in range(njt):
    fig, ax = plotPopulation(time_vect, mc.ffwd_p[i], mc.ffwd_n[i], "Motor cortex feedforward, "+lgd[i] )
    fgv.append(fig)
    axv.append(ax)
    max_y[i*3,:] = plt.gca().get_ylim()
    fig, ax = plotPopulation(time_vect, mc.fbk_p[i], mc.fbk_n[i], "Motor cortex feedback, "+lgd[i] )
    fgv.append(fig)
    axv.append(ax)
    max_y[i*3+1,:] = plt.gca().get_ylim()
    fig, ax = plotPopulation(time_vect, mc.out_p[i], mc.out_n[i], "Motor cortex output, "+lgd[i] )
    fgv.append(fig)
    axv.append(ax)
    max_y[i*3+2,:] = plt.gca().get_ylim()

max_y=np.max(max_y[:,1])
for i in range(njt):
    axv[i*3+0][1].set_ylim(top=max_y)
    axv[i*3+1][1].set_ylim(top=max_y)
    axv[i*3+2][1].set_ylim(top=max_y)
    if flagSaveFig:
        fgv[i*3+0].savefig(figPath+"motctx_ffwd_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".png",format="png")
        fgv[i*3+0].savefig(figPath+"motctx_ffwd_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".svg",format="svg")
        fgv[i*3+1].savefig(figPath+"motctx_fbk_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".png",format="png")
        fgv[i*3+1].savefig(figPath+"motctx_fbk_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".svg",format="svg")
        fgv[i*3+2].savefig(figPath+"motctx_out_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".png",format="png")
        fgv[i*3+2].savefig(figPath+"motctx_out_j"+str(i)+"_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".svg",format="svg")

# Feedforward and output populations (same scale)
fig, ax = plt.subplots(2,1)
for i in range(njt):
    mc.ffwd_p[i].plot_rate(time_vect,ax=ax[i],bar=False,color='r',label='ffwd')
    mc.ffwd_n[i].plot_rate(time_vect,ax=ax[i],bar=False,title=lgd[i]+" (Hz)",color='b')
    mc.out_p[i].plot_rate(time_vect,ax=ax[i],bar=False,color='r',linestyle=':',label='out')
    mc.out_n[i].plot_rate(time_vect,ax=ax[i],bar=False,title=lgd[i]+" (Hz)",color='b',linestyle=':')
    ax[i].set(ylim=(0, max_y))

ax[0].set_title('spike rate')
plt.legend()

if flagSaveFig:
    plt.savefig(figPath+"motctx_rates_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".png",format="png")
    plt.savefig(figPath+"motctx_rates_i_"+str(init_pos)+"_tgt_"+str(tgt_pos)+"_real_"+str(tgt_real)+".svg",format="svg")

plt.show()
