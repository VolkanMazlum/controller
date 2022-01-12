#!/usr/bin/env python3

import nest
import numpy as np
import matplotlib.pyplot as plt
import trajectories as tj
import time

from stateestimator import StateEstimator
from population_view import plotPopulation
from pointMass import PointMass

nest.Install("util_neurons_module")
res = nest.GetKernelStatus("resolution")

# Randomize
msd = int( time.time() * 1000.0 )
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})

flagSaveFig = False
figPath = './fig/stest/'
pthDat = "./data/"


# Dynamical system
m          = 2.0
ptMass     = PointMass(mass=m)
njt        = ptMass.numVariables()

# Neuron neurons
N = 50

# Time span and time vector expressed in ms (as all the rest in NEST)
time_span = 1000.0
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)


### STATE ESTIMATOR

kpred = 0.5
ksens = 0.5

param_se = {
    "pred_base_rate": 0.0,  # Prediction neurons (receive sensory prediction)
    "pred_kp":        1.0,
    "sens_base_rate": 0.0,  # Feedback neurons (receive sensory feedback)
    "sens_kp":        1.0,
    "out_base_rate":  0.0,   # Summation neurons
    "out_kp":         1.0,
    "wgt_scale":      1.0,   # Scale of connection weight from input to output populations (must be positive)
    "buf_sz":       100.0    # Size of the buffer to compute spike rate in basic_neurons (ms)
    }

se = StateEstimator(N, time_vect, ptMass, kpred=kpred, ksens=ksens, pathData=pthDat, **param_se)


##### TESTING POPULATIONS (predicted and actual sensory feedback)
# I am testing only the positive populations for simplicity

pred_p = nest.Create("poisson_generator",N)
nest.SetStatus(pred_p, {"rate": 10.0})

sens_p = nest.Create("poisson_generator",N)
nest.SetStatus(sens_p, {"rate": 20.0})

for i in range(njt):
    nest.Connect(pred_p,se.pred_p[i].pop, "one_to_one", syn_spec={'weight': 1.0})
    nest.Connect(sens_p,se.sens_p[i].pop, "one_to_one", syn_spec={'weight': 1.0})


##########################
nest.Simulate(time_span)


###### PLOTTING
lgd = ['x','y']
for i in range(njt):
    plotPopulation(time_vect, se.out_p[i], se.out_n[i], lgd[i])

plt.show()
