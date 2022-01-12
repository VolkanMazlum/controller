#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

# Just to get the following imports right!
sys.path.insert(1, '../')
import trajectories as tj
from sensoryneuron import SensoryNeuron

# Time span and time vector expressed in s
res = 0.0001
time_span = 1.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)
time_n    = len(time_vect)

# Poitions expressed in meters (mks system)
init_pos  = np.array([0.0])
final_pos = np.array([-5.0])
trj, pol  = tj.minimumJerk(init_pos, final_pos, time_vect)

N=50
idSt = 0
sn = SensoryNeuron(N, pos=False, idStart=idSt, bas_rate=0.0, kp=10.0)

for i in range(time_n):
    sn.update(trj[i], res, i)

evs, ts = sn.get_events()

fig, ax = plt.subplots(2,1,sharex=True)
ax[1].scatter(ts*res,evs,color='r',marker='.', s=1)
ax[0].plot(time_vect,trj)
plt.show()
