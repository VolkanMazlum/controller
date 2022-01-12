#!/usr/bin/env python3

import nest
import numpy as np
import matplotlib.pyplot as plt
import trajectories as tj

from cerebellum import Cerebellum
from pointMass import PointMass
from population_view import plotPopulation
import os
import mpi4py
import random

nest.Install("util_neurons_module")
nest.Install("cerebmodule")
nest.ResetKernel()
res = nest.GetKernelStatus("resolution")

def remove_files(pathData):
    # Remove existing .dat and/or .gdf files generated in previous simulations
    for f in os.listdir(pathData):
        if '.gdf' in f or '.dat' in f:
            os.remove(pathData+f)

###### CEREBELLUM ######
filename_h5 = "/home/benedetta/control_loops/hdf5/300x_200z_claudia_dcn_test_3.hdf5"
filename_config = '/home/benedetta/control_loops/configurations/mouse_cerebellum_cortex_update_dcn_copy_post_stepwise_colonna_R.json'
cerebellum = Cerebellum(filename_h5, filename_config, 1, "test")

###### Configure simulation parameters ######
flagSaveFig = False
SCATTER_PLOT = True
figPath = './fig/cerebellum/'
pthDat = "./data/cerebellum/"
if mpi4py.MPI.COMM_WORLD.rank == 0:
    remove_files(pthDat)

## Simulation ##
trial_len = 10
stop_len = 0
total_len = trial_len + stop_len
n_trial = 1

nest.SetKernelStatus({"overwrite_files": True,"data_path": pthDat})
if mpi4py.MPI.COMM_WORLD.rank == 0:
    print('Simulating {} ms'.format(total_len))
mpi4py.MPI.COMM_WORLD.Barrier()
nest.Simulate(total_len)

######### Collapsing data files into one file #########
tags = ['test']
names = []
network_neurons = ["Input inferior Olive pos","Input inferior Olive neg","Feedback pos","Feedback neg","State estimator pos","State estimator neg","Planner pos","Planner neg","Feedback cerebellum pos","Feedback cerebellum neg","mc_out_p_0","mc_out_n_0","mc_out_p_1","mc_out_n_1","sens_fbk_0_p","sens_fbk_0_n","sens_fbk_1_p","sens_fbk_1_n","Cereb pred pos","Cereb pred neg","State estimator Max pos","State estimator Max neg","Feedback smoothed pos","Feedback smoothed neg"]
cereb_neurons = ["granule_cell","golgi_cell","dcn_cell_glut_large","purkinje_cell","basket_cell","stellate_cell","dcn_cell_GABA","mossy_fibers",'io_cell',"glomerulus","dcn_cell_Gly-I"]

for t in tags:
    for n in cereb_neurons:
        names.append(t + '_' + n)
    #print(names.append(t + '_' + n))
names.extend(network_neurons)

#names = ["granule_cell","golgi_cell","dcn_cell_glut_large","purkinje_cell","basket_cell","stellate_cell","dcn_cell_GABA","mossy_fibers",'io_cell',"glomerulus","dcn_cell_Gly-I"]
files = [f for f in os.listdir(pthDat) if os.path.isfile(os.path.join(pthDat,f))]

if mpi4py.MPI.COMM_WORLD.rank == 0:
    file_list = []
    for name in names:
        if (name + '_spikes.gdf' not in files):
            for f in files:
                if (f.startswith(name)):
                    file_list.append(f)
            print(file_list)
            with open(pthDat + name + ".gdf", "w") as wfd:
                for f in file_list:
                    with open(pthDat + f, "r") as fd:
                        wfd.write(fd.read())
            for f in file_list:
                os.remove(pthDat+f)
            file_list = []
        else:
            print('Già fatto')
    print('Collapsing files ended')

# ########################### PLOTTING ###########################

# cell_numerosity = {
#     "granule_cell": len(cerebellum.S_GR),
#     "golgi_cell": len(cerebellum.S_Go),
#     "dcn_cell_glut_large": len(cerebellum.S_DCN),
#     "purkinje_cell": len(cerebellum.S_PC),
#     "basket_cell": len(cerebellum.S_BC),
#     "stellate_cell": len(cerebellum.S_SC),
#     "dcn_cell_GABA": len(cerebellum.S_DCN_GABA),
#     "mossy_fibers": len(cerebellum.S_Mf),
#     'io_cell': len(cerebellum.S_IO),
# }

# if mpi4py.MPI.COMM_WORLD.rank == 0:
#     print('Start reading data')
#     files = [f for f in os.listdir(pthDat) if os.path.isfile(os.path.join(pthDat,f))]
#     IDs = {}
#     SD = {}
#     times = {}
#     for cell in names:
#         print('Reading:',cell)
#         for f in files:
#             if f.startswith(cell):
#                 break
#         cell_f = open(pthDat+f,'r').read()
#         cell_f = cell_f.split('\n')
#         IDs[cell] = {}
#         SD[cell] = {'evs': [], 'ts': []}
#         for i in range(len(cell_f)-1):
#             splitted_string = cell_f[i].split('\t')
#             ID_cell = float(splitted_string[0])
#             time_cell = float(splitted_string[1])
#             SD[cell]['evs'].append(ID_cell)
#             SD[cell]['ts'].append(time_cell)
#             if str(ID_cell) in IDs[cell].keys():
#                 IDs[cell][str(ID_cell)].append(time_cell)
#             else:
#                 IDs[cell][str(ID_cell)] = [time_cell]

#     print('Start making plots')
#     for name_id,cell in enumerate(names):
#         if cell in ["golgi_cell","glomerulus","dcn_cell_Gly-I"]:
#             continue
#         if (IDs[cell].keys()):
#             beginning = 0
#             bin_duration = 10
#             if cell in ["dcn_cell_glut_large","purkinje_cell","basket_cell","stellate_cell","io_cell"]:
#                 freq_pos = []
#                 freq_neg = []
#                 plt.figure(figsize=(10,8))
#                 for start in range(beginning, total_len*n_trial, bin_duration):
#                     n_spikes_pos = 0
#                     n_spikes_neg = 0
#                     end = start + bin_duration
#                     for key in IDs[cell].keys():
#                         times = [i for i in IDs[cell][key] if i>=start and i< end]
#                         if float(key) in cerebellum.Nest_ids[cell]["positive"]:
#                             n_spikes_pos += len(times)
#                         elif float(key) in cerebellum.Nest_ids[cell]["negative"]:
#                             n_spikes_neg += len(times)
#                         else:
#                             print(d)
#                             pass
#                     freq_bin_pos = n_spikes_pos/(bin_duration/1000*len(cerebellum.Nest_ids[cell]["positive"]))
#                     freq_bin_neg = n_spikes_neg/(bin_duration/1000*len(cerebellum.Nest_ids[cell]["negative"]))
#                     freq_pos.append(freq_bin_pos)
#                     freq_neg.append(freq_bin_neg)
#                 x = range(beginning, total_len*n_trial, bin_duration)
#                 plt.plot(x,freq_pos,'b', label='positive')
#                 plt.plot(x,freq_neg,'r', label='negative')
#                 plt.title('Spike frequency ' + names[name_id], size =25)
#                 plt.xlabel('Time [ms]', size =25)
#                 plt.ylabel('Frequency [Hz]', size =25)
#                 plt.xlim(0,total_len*n_trial)
#                 plt.xticks(fontsize=25)
#                 plt.yticks(fontsize=25)

#                 start = 0
#                 end = total_len*n_trial
#                 n_spikes_pos = 0
#                 n_spikes_neg = 0
#                 for key in IDs[cell].keys():
#                     times = [i for i in IDs[cell][key] if i>=start and i< end]
#                     if float(key) in cerebellum.Nest_ids[cell]["positive"]:
#                         n_spikes_pos += len(times)
#                     elif float(key) in cerebellum.Nest_ids[cell]["negative"]:
#                         n_spikes_neg += len(times)
#                     else:
#                         print('STRANO')
#                         pass
#                 mean_freq_pos = n_spikes_pos/((end-start)/1000*len(cerebellum.Nest_ids[cell]["positive"]))
#                 mean_freq_neg = n_spikes_neg/((end-start)/1000*len(cerebellum.Nest_ids[cell]["negative"]))
#                 x = [start,end]
#                 y = [mean_freq_pos]*len(x)
#                 plt.plot(x,y,'b',linewidth = 3)
#                 y = [mean_freq_neg]*len(x)
#                 plt.plot(x,y,'r',linewidth = 3)
#                 plt.legend()
#                 plt.savefig(figPath+'Spike frequency ' + names[name_id]+'.svg')

#             else:
#                 freq = []
#                 plt.figure(figsize=(10,8))
#                 for start in range(beginning, total_len*n_trial, bin_duration):
#                     n_spikes = 0
#                     end = start + bin_duration
#                     for key in IDs[cell].keys():
#                         times = [i for i in IDs[cell][key] if i>=start and i< end]
#                         n_spikes += len(times)
#                     freq_bin = n_spikes/(bin_duration/1000*cell_numerosity[cell])
#                     freq.append(freq_bin)
#                 x = range(beginning, total_len*n_trial, bin_duration)
#                 plt.plot(x,freq)
#                 plt.title('Spike frequency ' + names[name_id], size =25)
#                 plt.xlabel('Time [ms]', size =25)
#                 plt.ylabel('Frequency [Hz]', size =25)
#                 plt.xlim(0,total_len*n_trial)
#                 plt.xticks(fontsize=25)
#                 plt.yticks(fontsize=25)
#                 start = 0
#                 end = total_len*n_trial
#                 n_spikes = 0
#                 for key in IDs[cell].keys():
#                     times = [i for i in IDs[cell][key] if i>=start and i< end]
#                     n_spikes += len(times)
#                 mean_freq = n_spikes/((end-start)/1000*cell_numerosity[cell])
#                 x = [start,end]
#                 y = [mean_freq]*len(x)
#                 plt.plot(x,y,'r',linewidth = 3)
#                 plt.savefig(figPath+'Spike frequency ' + names[name_id]+'.svg')

#             if SCATTER_PLOT:
#                 plt.figure(figsize=(10,8))
#                 y_min = np.min(SD[cell]['evs'])
#                 plt.scatter(SD[cell]['ts'], SD[cell]['evs']-y_min, marker='.', s = 200)
#                 plt.title('Scatter plot '+ names[name_id]+ ' neurons', size =25)
#                 plt.xlabel('Time [ms]', size =25)
#                 plt.ylabel('Neuron ID', size =25)
#                 plt.xlim(0,total_len*n_trial)
#                 plt.xticks(fontsize=25)
#                 plt.yticks(fontsize=25)
#                 plt.savefig(figPath+'Scatter plot '+ names[name_id]+ ' neurons.svg')

#         else:
#             print('Population '+cell+ ' is NOT spiking')

#     plt.show()
