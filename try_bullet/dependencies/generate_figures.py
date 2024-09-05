import numpy as np
import matplotlib.pyplot as plt
import os
from population_view import plotPopulation_sd, computePSTH_sd
from settings import Simulation, Experiment
import json

pathFig = '/dependencies/fig/'
pathData = '/dependencies/data/'
saveFig=True
# Load reference files

trj = np.loadtxt(os.getcwd() + pathData+ 'trajectory.txt')
mtCmds = np.loadtxt(os.getcwd() + pathData+ 'motor_commands.txt')

# Load recorded spikes
with open('spike_recorders.json', 'r') as file:
    data = json.load(file)

planner_times_pos = data['planner']['pos']['times']
planner_senders_pos = data['planner']['pos']['senders']
planner_times_neg = data['planner']['neg']['times']
planner_senders_neg = data['planner']['neg']['senders']

ffwd_times_pos = data['motor_cortex']['ffwd']['pos']['times']
ffwd_senders_pos = data['motor_cortex']['ffwd']['pos']['senders']
ffwd_times_neg = data['motor_cortex']['ffwd']['neg']['times']
ffwd_senders_neg = data['motor_cortex']['ffwd']['neg']['senders']

fbk_times_pos = data['motor_cortex']['fbk']['pos']['times']
fbk_senders_pos = data['motor_cortex']['fbk']['pos']['senders']
fbk_times_neg = data['motor_cortex']['fbk']['neg']['times']
fbk_senders_neg = data['motor_cortex']['fbk']['neg']['senders']

out_times_pos = data['motor_cortex']['out']['pos']['times']
out_senders_pos = data['motor_cortex']['out']['pos']['senders']
out_times_neg = data['motor_cortex']['out']['neg']['times']
out_senders_neg = data['motor_cortex']['out']['neg']['senders']

state_times_pos = data['state']['pos']['times']
state_senders_pos = data['state']['pos']['senders']
state_times_neg = data['state']['neg']['times']
state_senders_neg = data['state']['neg']['senders']

brainstem_times_pos = data['brainstem']['pos']['times']
brainstem_senders_pos = data['brainstem']['pos']['senders']
brainstem_times_neg = data['brainstem']['neg']['times']
brainstem_senders_neg = data['brainstem']['neg']['senders']

sensory_times_pos = data['sensory']['pos']['times']
sensory_senders_pos = data['sensory']['pos']['senders']
sensory_times_neg = data['sensory']['neg']['times']
sensory_senders_neg = data['sensory']['neg']['senders']

prediction_times_pos = data['prediction']['pos']['times']
prediction_senders_pos = data['prediction']['pos']['senders']
prediction_times_neg = data['prediction']['neg']['times']
prediction_senders_neg = data['prediction']['neg']['senders']

# Time variables
res_sim = 20 #[ms]
res = 0.1
time_span = 1000.0 #[ms]
n_trial = 1
time_vect  = np.linspace(0, time_span, num=int(np.round(time_span/res_sim)), endpoint=True)
time_reference = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)
N = 50

# Plot figures
# Planner + trajectory
lgd = ['theta']
time_vect_paused = time_vect

reference =[trj]
legend = ['trajectory']
styles=['k']
time_vecs=[time_reference]

plotPopulation_sd(time_vect_paused, N, planner_times_pos, planner_senders_pos, planner_times_neg, planner_senders_neg, reference, time_vecs,legend, styles, title=lgd[0],buffer_size=15)
plt.suptitle("Planner")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+"planner_"+lgd[0]+".png")

### Motor cortex
# Feedforward
reference =[mtCmds]
legend = ['motor commands']

plotPopulation_sd(time_vect_paused, N, ffwd_times_pos, ffwd_senders_pos, ffwd_times_neg, ffwd_senders_neg, reference, time_vecs,legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("Mc ffwd")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+ "mc_ffwd_"+lgd[0]+".png")

# Feedback
#bins_p,count_p,rate_p = planner.pops_p[0].computePSTH(time_vect_paused, 15)
bins_p,count_p,rate_p = computePSTH_sd(time_vect_paused, N, planner_times_pos,  15)
bins_n,count_n,rate_n = computePSTH_sd(time_vect_paused, N, planner_times_neg,  15)
#bins_stEst_p,count_stEst_p,rate_stEst_p = stEst.pops_p[0].computePSTH(time_vect_paused, 15)
bins_stEst_p,count_stEst_p,rate_stEst_p = computePSTH_sd(time_vect_paused, N, state_times_pos,  15)
bins_stEst_n,count_stEst_n,rate_stEst_n = computePSTH_sd(time_vect_paused, N, state_times_neg,  15)

reference =[rate_p-rate_stEst_p, rate_n - rate_stEst_n]
time_vecs = [bins_p[:-1], bins_n[:-1]]
legend = ['diff_p', 'diff_n']
styles = ['r--', 'b--']

plotPopulation_sd(time_vect_paused, N, fbk_times_pos, fbk_senders_pos, fbk_times_neg, fbk_senders_neg, reference, time_vecs, legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("Mc fbk")
if saveFig:
   plt.savefig(os.getcwd()+ pathFig+"mc_fbk_"+lgd[0]+".png")

# Out
bins_p,count_p,rate_p = computePSTH_sd(time_vect_paused, N, ffwd_times_pos,  15)
bins_n,count_n,rate_n = computePSTH_sd(time_vect_paused, N, ffwd_times_neg,  15)
bins_fbk_p,count_fbk_p,rate_fbk_p = computePSTH_sd(time_vect_paused, N, fbk_times_pos,  15)
bins_fbk_n,count_fbk_n,rate_fbk_n = computePSTH_sd(time_vect_paused, N, fbk_times_neg,  15)

reference =[rate_p+rate_fbk_p, rate_n + rate_fbk_n]
time_vecs = [bins_p[:-1], bins_n[:-1]]
legend = ['sum_p', 'sum_n']
styles = ['r--', 'b--']

plotPopulation_sd(time_vect_paused, N, out_times_pos, out_senders_pos, out_times_neg, out_senders_neg, reference, time_vecs, legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("Mc out")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+"mc_out_"+lgd[0]+".png")

# Brainstem
bins_p,count_p,rate_p = computePSTH_sd(time_vect_paused, N, out_times_pos,  15)
bins_n,count_n,rate_n = computePSTH_sd(time_vect_paused, N, out_times_neg,  15)

reference =[rate_p, rate_n]
time_vecs = [bins_p[:-1], bins_n[:-1]]
legend = ['out_p', 'out_n']
styles = ['r', 'b']

plotPopulation_sd(time_vect_paused, N, brainstem_times_pos, brainstem_senders_pos, brainstem_times_neg, brainstem_senders_neg, reference, time_vecs, legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("Mc out")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+"_brain_stem_"+lgd[0]+".png")

reference =[]
time_vecs = []
legend = []
styles = []

plotPopulation_sd(time_vect_paused, N, sensory_times_pos, sensory_senders_pos, sensory_times_neg, sensory_senders_neg, reference, time_vecs, legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("Sensory")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+"_sensory_"+lgd[0]+".png")

bins_p,count_p,rate_p = computePSTH_sd(time_vect_paused, N, sensory_times_pos,  15)
bins_n,count_n,rate_n = computePSTH_sd(time_vect_paused, N, sensory_times_neg,  15)
bins_pred_p,count_pred_p,rate_pred_p = computePSTH_sd(time_vect_paused, N, prediction_times_pos,  15)
bins_pred_n,count_pred_n,rate_pred_n = computePSTH_sd(time_vect_paused, N, prediction_times_neg,  15)

reference =[rate_p-rate_n, rate_pred_p - rate_pred_n]
time_vecs = [bins_p[:-1], bins_n[:-1]]
legend = ['net_sensory', 'net_prediction']
styles = ['g--', 'r--']

plotPopulation_sd(time_vect_paused, N, state_times_pos, state_senders_pos, state_times_neg, state_senders_neg, reference, time_vecs, legend, styles,title=lgd[0],buffer_size=15)
plt.suptitle("State")
if saveFig:
    plt.savefig(os.getcwd()+pathFig+"_state_"+lgd[0]+".png")


