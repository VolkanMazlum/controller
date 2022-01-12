#!/usr/bin/env python3

import sys
import music
import numpy as np
import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../')
import trajectories as tj
from sensoryneuron import SensoryNeuron
#from population_view import plotPopulation

import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

############################ GENERAL CONFIG ############################

pth  = './data/'   # Where to save the files

njt  = 2 # This will come from the plant object
N    = 20 # Number of sensory neurons (per subpopulation)
idSt = 0  # First ID

# Time span and time vector expressed in s
res = 0.0001
time_span = 1.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)
time_n    = len(time_vect)

# Poitions expressed in meters (mks system)
init_pos  = np.array([0.0, 0.0])
final_pos = np.array([2.0, 5.0])
trj, pol  = tj.minimumJerk(init_pos, final_pos, time_vect)

############################ MUSIC CONFIG ############################

firstId = 0       # First neuron taken care of by this MPI rank
size    = 2*N*njt # Number of neurons taken care of by this MPI rank (N pos/neg per joint)

# The MUSIC setup object is used to configure the simulation
setup = music.Setup()

# Creation of MUSIC ports
outp = setup.publishEventOutput("out")

# Configuration of the output port
outp.map (music.Index.GLOBAL,
          base=firstId,
          size=size)


##################### NETWORK #####################

sn_p = [] # Positive sensory neurons
sn_n = [] # Negative sensory neurons
for i in range(njt):
    # Positive
    idSt_p = idSt+2*N*i
    tmp    = SensoryNeuron(N, pos=True, idStart=idSt_p, bas_rate=0.0, kp=10.0)
    tmp.connect(outp)   # Connect to output port
    sn_p.append(tmp)
    # Negative
    idSt_n = idSt_p+N
    tmp    = SensoryNeuron(N, pos=False, idStart=idSt_n, bas_rate=0.0, kp=10.0)
    tmp.connect(outp)   # Connect to output port
    sn_n.append(tmp)


##################### SIMULATION #####################

# Start the runtime phase
runtime = music.Runtime(setup, res)

it=0
tickt = runtime.time()
while tickt < time_span:

    # Update neuron states and send spikes
    for j in range(njt):
        sn_p[j].update(trj[it,j], res, tickt)
        sn_n[j].update(trj[it,j], res, tickt)

    runtime.tick()
    tickt = runtime.time()
    it=it+1

runtime.finalize()


################## PLOT AND SAVE ##################

lgd = ['x','y']

plt.figure()
plt.plot(time_vect,trj)
plt.legend(lgd)
plt.xlabel('time (s)')
plt.ylabel('position (m)')

# for i in range(njt):
#     plotPopulation(time_vect, sn_p[i], sn_n[i], title=lgd[i], buffer_size=0.015)

plt.show()

# Save into file
for i in range(njt):
    np.savetxt(pth+"spike_send_p_"+str(i)+".csv",np.array(sn_p[i].spike),delimiter=',')
    np.savetxt(pth+"spike_send_n_"+str(i)+".csv",np.array(sn_n[i].spike),delimiter=',')
