import numpy as np
import nest
import os
import matplotlib.pyplot as plt
# Adjust env vars to be able to import the NESTML-generated module
ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
new_path = ld_lib_path + ":"+"../nestml/target"
os.environ['LD_LIBRARY_PATH'] = new_path

import json

####### NETWORK SETUP
# Load parameters from file
f = open('new_params.json')
params = json.load(f)
f.close()

mc_params = params["modules"]["motor_cortex"]
plan_params = params["modules"]["planner"]
pops_params = params["pops"]
conn_params = params["connections"]

res = 0.1 #[ms]
time_span = 500.0 #[ms]
n_trial = 1
time_vect  = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

njt = 1
trj = np.loadtxt('trajectory.txt')
motorCommands=np.loadtxt('motor_commands.txt')

N = 50 # Number of neurons for each (sub-)population
nTot = 2*N*njt # Total number of neurons (one positive and one negative population for each DOF)

nest.ResetKernel()
nest.SetKernelStatus({"resolution": res})

### Install NESTML-generated module
nest.Install("controller_module")

### PLANNER: create and initialize the Planner population
# Input: target --> minimum jerk trajectory
# Output: spikes (firing rate proportional to elbow angle)
for i in range(njt):
    planner_p = nest.Create("tracking_neuron_nestml", n=N, params={"kp": plan_params["kp"], "base_rate": plan_params["base_rate"], "pos": True, "traj": trj, "simulation_steps": len(trj)})

    planner_n = nest.Create("tracking_neuron_nestml", n=N, params={"kp": plan_params["kp"], "base_rate": plan_params["base_rate"], "pos": False, "traj": trj, "simulation_steps": len(trj)})


### FEEDFORWARD MOTOR CORTEX: create and initialize the Motor cortex populationù
# Input: target --> double derivative + inverse dynamics --> torque to be applied to elbow joint (motor command)
# Output: spikes (firing rate proportional to torque)
for i in range(njt):
    motor_p = nest.Create("tracking_neuron_nestml", n=N, params={"kp":mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': True, 'traj': motorCommands, 'simulation_steps': len(motorCommands)})

    motor_n = nest.Create("tracking_neuron_nestml", n=N, params={'kp':mc_params["ffwd_kp"], 'base_rate':mc_params['ffwd_base_rate'], 'pos': False, 'traj': motorCommands, 'simulation_steps': len(motorCommands)})

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
def computePSTH(time, ts, evs, buffer_sz=10):
        t_init = time[0]
        t_end  = time[ len(time)-1 ]
        count, bins = np.histogram( ts, bins=np.arange(t_init,t_end+1,buffer_sz) )
        rate = 1000*count/(N*buffer_sz)
        return bins, count, rate
        
def plot_rate(time, ts, evs, buffer_sz=10, title='', ax=None, bar=True, **kwargs):

        t_init = time[0]
        t_end  = time[ len(time)-1 ]

        bins,count,rate = computePSTH(time, ts, evs,buffer_sz)
        rate_sm = np.convolve(rate, np.ones(5)/5,mode='same')

        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots(1)

        if bar:
            ax.bar(bins[:-1], rate, width=bins[1]-bins[0],**kwargs)
            ax.plot(bins[:-1],rate_sm,color='k')
        else:
            ax.plot(bins[:-1],rate_sm,**kwargs)
        ax.set(xlim=(t_init, t_end))
        ax.set_ylabel(title)
        
        
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
   plot_rate(time_vect, ts_p, y_p,ax=ax[1],bar=True,color='r',label='brainstem')
   plot_rate(time_vect, ts_n, y_n,ax=ax[1],bar=True,color='b',label='brainstem')
   plt.savefig("planner.png")

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
   plot_rate(time_vect, ts_p, y_p,ax=ax[1],bar=True,color='r',label='brainstem')
   plot_rate(time_vect, ts_n, y_n,ax=ax[1],bar=True,color='b',label='brainstem')
plt.savefig("mc_ffw.png")

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
   plot_rate(time_vect, ts_p, y_p,ax=ax[1],bar=True,color='r',label='brainstem')
   plot_rate(time_vect, ts_n, y_n,ax=ax[1],bar=True,color='b',label='brainstem')
plt.savefig("brainstem.png")


########## COMPUTE OUTPUT (net firing rate)
   


