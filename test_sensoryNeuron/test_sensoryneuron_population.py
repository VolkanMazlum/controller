#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../')
import trajectories as tj
from sensoryneuron import SensoryNeuron
from population_view import plotPopulation


############################ GENERAL CONFIG ############################

njt  = 2 # This will come from the plant object

N    = 50 # Number of sensory neurons (per subpopulation)
idSt = 0  # First ID

# Time span and time vector expressed in s
res = 0.0001
time_span = 1.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)
time_n    = len(time_vect)

# Poitions expressed in meters (mks system)
init_pos  = np.array([-2.0, -2.0])
final_pos = np.array([2.0, 5.0])
trj, pol  = tj.minimumJerk(init_pos, final_pos, time_vect)

##################### NETWORK #####################

sn_p = [] # Positive sensory neurons
sn_n = [] # Negative sensory neurons
for i in range(njt):
    # Positive
    idSt_p = idSt+2*N*i
    tmp    = SensoryNeuron(N, pos=True, idStart=idSt_p, bas_rate=0.0, kp=10.0)
    sn_p.append(tmp)
    # Negative
    idSt_n = idSt_p+N
    tmp    = SensoryNeuron(N, pos=False, idStart=idSt_n, bas_rate=0.0, kp=10.0)
    sn_n.append(tmp)


##################### SIMULATION #####################

# Update neuron states and send spikes
for it in range(time_n):
    for j in range(njt):
        sn_p[j].update(trj[it,j], res, time_vect[it])
        sn_n[j].update(trj[it,j], res, time_vect[it])


################## PLOT AND SAVE ##################

lgd = ['x','y']

plt.figure()
plt.plot(time_vect,trj)
plt.legend(lgd)
plt.xlabel('time (s)')
plt.ylabel('position (m)')

for i in range(njt):
    plotPopulation(time_vect, sn_p[i], sn_n[i], title=lgd[i], buffer_size=0.015)

plt.show()
