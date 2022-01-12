#!/usr/bin/env python3

import sys
import nest
import music
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../')
from population_view import plotPopulation, PopView


############################ GENERAL CONFIG ############################

pth  = './data/'   # Where to save the files

njt  = 2           # Number of variables to be controlled (it will come from the plant object)
N    = 20          # Number of neurons per subpopulation
nTot = 2*N*njt     # Total number of neurons

res = 0.1          # Resolution (ms)
time_span = 1000.0 # Simulation time (ms)
time_vect = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

delay_fbk = 0.1    # Delay sensory feedback. It cannot be < resolution

nest.SetKernelStatus({"overwrite_files": True})
nest.SetKernelStatus({"resolution": res})


########################### RECEIVING NETWORK

sn_p=[]
sn_n=[]
for j in range(njt):
    # Positive neurons
    tmp_p = nest.Create ("parrot_neuron", N)
    sn_p.append( PopView(tmp_p, time_vect, to_file=True, label=pth+'spk_rec_p_'+str(j)) )
    # Negative neurons
    tmp_n = nest.Create ("parrot_neuron", N)
    sn_n.append( PopView(tmp_n, time_vect, to_file=True, label=pth+'spk_rec_n_'+str(j)) )


########################### INPUT

# Creation of MUSIC input port with N input channels
meip = nest.Create ('music_event_in_proxy', nTot, params = {'port_name': 'spikes_in'})
for i, n in enumerate(meip):
    nest.SetStatus([n], {'music_channel': i})

# Divide channels based on function (using channel order)
for j in range(njt):
    #### Positive channels
    idxSt_p = 2*N*j
    idxEd_p = idxSt_p + N
    nest.Connect( meip[idxSt_p:idxEd_p], sn_p[j].pop, 'one_to_one', {"weight":1.0, "delay": delay_fbk} )
    #### Negative channels
    idxSt_n = idxEd_p
    idxEd_n = idxSt_n + N
    nest.Connect( meip[idxSt_n:idxEd_n], sn_n[j].pop, 'one_to_one', {"weight":1.0, "delay": delay_fbk} )


########################## SIMULATE

nest.Simulate (time_span)


###################### PLOT ##########################

lgd = ['x','y']

for j in range(njt):
    plotPopulation(time_vect, sn_p[j], sn_n[j], title=lgd[j], buffer_size=15)
