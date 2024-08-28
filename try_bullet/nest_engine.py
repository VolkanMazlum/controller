import numpy as np
import time
import os
import nest
import json

from motorcortex import MotorCortex
from planner import Planner
from stateestimator import StateEstimator, StateEstimator_mass

bd = nrp_core.engines.nest_server.brain_devices


# Opening JSON file to get parameters
f = open('dependencies/new_params.json')
params = json.load(f)
f.close()

print(params.keys())
mc_params = params["modules"]["motor_cortex"]
plan_params = params["modules"]["planner"]
spine_params = params["modules"]["spine"]
state_params = params["modules"]["state"]
state_se_params = params["modules"]["state_se"]
pops_params = params["pops"]
conn_params = params["connections"]

#%%  SIMULATION
res = 0.1 #[ms]
time_span = 1000.0 #[ms]
n_trial = 1
time_vect  = np.linspace(0, time_span, num=int(np.round(time_span/res)), endpoint=True)

njt = 1
trj = np.loadtxt('dependencies/trajectory.txt')
motorCommands=np.loadtxt('dependencies/motor_commands.txt')
assert len(trj) == len(time_vect), "Arrays do not have the same length"
assert len(motorCommands) == len(time_vect), "Arrays do not have the same length"
N = 25 # Number of neurons for each (sub-)population
nTot = 2*N*njt # Total number of neurons (one positive and one negative population for each DOF)

nest.ResetKernel()
nest.SetKernelStatus({"resolution": res})

# # Cerebellum
cereb_controlled_joint = 0 # x = 0, y = 1

# Install the module containing neurons for planner and motor cortex
nest.Install("controller_module")

#### Planner
print("init planner")
planner = p.Planner(N, njt, time_vect, trj, plan_params["kpl"], plan_params["base_rate"], plan_params["kp"])

#### Motor cortex
print("init mc")
preciseControl = False # Precise or approximated ffwd commands?
mc = mco.MotorCortex(N, njt, time_vect, motorCommands, **mc_params)

#### State Estimator
print("init state")
buf_sz = state_params['buffer_size']
additional_state_params = {'N_fbk': N, 'N_pred': N, 'fbk_bf_size': N*int(buf_sz/res), 'pred_bf_size': N*int(buf_sz/res)}
state_params.update(additional_state_params)
stEst = se.StateEstimator_mass(N, njt, time_vect, **state_params)

#%% SPINAL CORD ########################
delay_fbk          = params["modules"]["spine"]["fbk_delay"]
wgt_sensNeur_spine = params["modules"]["spine"]["wgt_sensNeur_spine"]

#### Sensory feedback (Parrot neurons on Sensory neurons) 
#### DA COLLEGARE CON I BD CON INFO DA BULLET
for j in range(njt):
    # Positive neurons
    sn_parrot_p = nest.Create ("parrot_neuron", N)
    # Negative neurons
    sn_parrot_n = nest.Create ("parrot_neuron", N)

sn_p = bd.PoissonSpikeGenerator(nest, sn_parrot_p, conn_spec='all_to_all')
sn_n = bd.PoissonSpikeGenerator(nest, sn_parrot_n, conn_spec='all_to_all')

#%% State estimator #######
# Scale the cerebellar prediction up to 1000 Hz
# in order to have firing rate suitable for the State estimator
# and all the other structures inside the control system

prediction_p = nest.Create("diff_neuron_nestml", N)
nest.SetStatus(prediction_p, {"kp": pops_params["prediction"]["kp"], "pos": True, "buffer_size": pops_params["prediction"]["buffer_size"], "base_rate": pops_params["prediction"]["base_rate"], "simulation_steps": len(time_vect)}) #5.5
prediction_n = nest.Create("diff_neuron_nestml", N)
nest.SetStatus(prediction_n, {"kp": pops_params["prediction"]["kp"], "pos": False, "buffer_size": pops_params["prediction"]["buffer_size"], "base_rate": pops_params["prediction"]["base_rate"], "simulation_steps": len(time_vect)}) #5.5

for j in range(njt):
    ''
    
    if j == cereb_controlled_joint:
        # Modify variability sensory feedback ("smoothed")
        fbk_smoothed_p = nest.Create("diff_neuron_nestml", N)
        nest.SetStatus(fbk_smoothed_p, {"kp": pops_params["fbk_smoothed"]["kp"], "pos": True, "buffer_size": pops_params["fbk_smoothed"]["buffer_size"], "base_rate": pops_params["fbk_smoothed"]["base_rate"], "simulation_steps": len(time_vect)})
        
        fbk_smoothed_n = nest.Create("diff_neuron_nestml", N)
        nest.SetStatus(fbk_smoothed_n, {"kp": pops_params["fbk_smoothed"]["kp"], "pos": False, "buffer_size": pops_params["fbk_smoothed"]["buffer_size"], "base_rate": pops_params["fbk_smoothed"]["base_rate"], "simulation_steps": len(time_vect)})
        
        nest.Connect(sn_parrot_p, fbk_smoothed_p, "all_to_all", syn_spec={"weight": conn_params["sn_fbk_smoothed"]["weight"], "delay": conn_params["sn_fbk_smoothed"]["delay"]})
        w = -conn_params["sn_fbk_smoothed"]["weight"]
        nest.Connect(sn_parrot_n, fbk_smoothed_n, "all_to_all", syn_spec={"weight": w, "delay": conn_params["sn_fbk_smoothed"]["delay"]})
        
        # Positive neurons
        '''
        for i, pre in enumerate(fbk_smoothed_p):
            for k, post in enumerate(stEst.pops_p):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": 1.0, "receptor_type": i + 1})
                
        
        for i, pre in enumerate(prediction_p):
            for k, post in enumerate(stEst.pops_p):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": 1.0, "receptor_type": i + 1 + N})
        '''
        for i in range(len(fbk_smoothed_p)):
            nest.Connect(fbk_smoothed_p[i], stEst.pops_p, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1})
        for i in range(len(prediction_p)):
            nest.Connect(prediction_p[i], stEst.pops_p, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1 + N})
        

        # Negative neurons
        '''
        for i, pre in enumerate(fbk_smoothed_n):
            for k, post in enumerate(stEst.pops_n):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": 1.0, "receptor_type": i + 1})

        for i, pre in enumerate(prediction_n):
            for k, post in enumerate(stEst.pops_n):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": 1.0, "receptor_type": i + 1 + N})
        '''
        for i in range(len(fbk_smoothed_n)):
            nest.Connect(fbk_smoothed_n[i], stEst.pops_n, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1})
        for i in range(len(prediction_n)):
            nest.Connect(prediction_n[i], stEst.pops_n, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1 + N})
        
    else:

        # Positive neurons
        #nest.Connect(sn_p[j].pop, stEst.pops_p[j].pop, "all_to_all", syn_spec=conn_params["sn_state"])
        '''
        for i, pre in enumerate(sn_parrot_p):
            for k, post in enumerate(stEst.pops_p):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": conn_params["sn_state"]["weight"], "receptor_type": i + 1})
        '''
        for i in range(len(sn_parrot_p)):
            nest.Connect(sn_parrot_p[i], stEst.pops_p, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1})
        # Negative neurons
        '''
        for i, pre in enumerate(sn_parrot_n):
            for k, post in enumerate(stEst.pops_n):
                nest.Connect(pre, post, "one_to_one", syn_spec = {"weight": conn_params["sn_state"]["weight"], "receptor_type": i + 1})
        '''
        for i in range(len(sn_parrot_n)):
            nest.Connect(sn_parrot_n[i], stEst.pops_n, "all_to_all", syn_spec = {"weight": 1.0, "receptor_type": i + 1})

print("init connections feedback")

#%% CONNECTIONS
#### Connection Planner - Motor Cortex feedback (excitatory)
wgt_plnr_mtxFbk   = conn_params["planner_mc_fbk"]["weight"]

# Delay between planner and motor cortex feedback.
# It needs to compensate for the delay introduced by the state estimator
#delay_plnr_mtxFbk = brain.stEst_param["buf_sz"] # USE THIS WITH REAL STATE ESTIMATOR
delay_plnr_mtxFbk = conn_params["planner_mc_fbk"]["delay"]                         # USE THIS WITH "FAKE" STATE ESTIMATOR

for j in range(njt):
    nest.Connect(planner.pops_p, mc.fbk_p, "one_to_one", syn_spec = {'weight': wgt_plnr_mtxFbk, 'delay': delay_plnr_mtxFbk})
    nest.Connect(planner.pops_p, mc.fbk_n, "one_to_one", syn_spec = {'weight': wgt_plnr_mtxFbk, 'delay': delay_plnr_mtxFbk})
    nest.Connect(planner.pops_n, mc.fbk_p, "one_to_one", syn_spec = {'weight': -wgt_plnr_mtxFbk, 'delay': delay_plnr_mtxFbk})
    nest.Connect(planner.pops_n, mc.fbk_n, "one_to_one", syn_spec = {'weight': -wgt_plnr_mtxFbk, 'delay': delay_plnr_mtxFbk})
    '''
    planner.pops_p[j].connect( mc.fbk_p[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
    planner.pops_p[j].connect( mc.fbk_n[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
    planner.pops_n[j].connect( mc.fbk_p[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
    planner.pops_n[j].connect( mc.fbk_n[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
    '''

#### Connection State Estimator - Motor Cortex feedback (Inhibitory)
wgt_stEst_mtxFbk = conn_params["state_mc_fbk"]["weight"]
'''
nest.Connect(mc.out_p[cereb_controlled_joint].pop, motor_commands_p, "all_to_all", syn_spec={"weight": conn_params["mc_out_motor_commands"]["weight"], "delay": conn_params["mc_out_motor_commands"]["delay"]})
nest.Connect(mc.out_n[cereb_controlled_joint].pop, motor_commands_n, "all_to_all", syn_spec={"weight": -conn_params["mc_out_motor_commands"]["weight"], "delay": conn_params["mc_out_motor_commands"]["delay"]})
''
nest.Connect(motor_commands_p, cerebellum_forw.Nest_Mf[-n_forw:], {'rule': 'one_to_one'},syn_spec={'weight':1.0})
nest.Connect(motor_commands_n, cerebellum_forw.Nest_Mf[0:n_forw], {'rule': 'one_to_one'},syn_spec={'weight':1.0})#TODO add weight

# Scale the feedback signal to 0-60 Hz in order to be suitable for the cerebellum
feedback_p = nest.Create("diff_neuron", N)
nest.SetStatus(feedback_p, {"kp": pops_params["feedback"]["kp"], "pos": True, "buffer_size": pops_params["feedback"]["buffer_size"], "base_rate": pops_params["feedback"]["base_rate"]})
feedback_n = nest.Create("diff_neuron", N)
nest.SetStatus(feedback_n, {"kp": pops_params["feedback"]["kp"], "pos": False, "buffer_size": pops_params["feedback"]["buffer_size"], "base_rate": pops_params["feedback"]["base_rate"]})

nest.Connect(sn_p[cereb_controlled_joint].pop, feedback_p, 'all_to_all', syn_spec={"weight": conn_params["sn_fbk_smoothed"]["weight"], "delay": conn_params["sn_fbk_smoothed"]["delay"]})
nest.Connect(sn_n[cereb_controlled_joint].pop, feedback_n, 'all_to_all', syn_spec={"weight": -conn_params["sn_fbk_smoothed"]["weight"], "delay": conn_params["sn_fbk_smoothed"]["delay"]})
'''

# Connect state estimator (bayesian) to the Motor Cortex
for j in range(njt):
    nest.Connect(stEst.pops_p,mc.fbk_p, "one_to_one", {"weight": wgt_stEst_mtxFbk, "delay": res})
    nest.Connect(stEst.pops_p,mc.fbk_n, "one_to_one", {"weight": wgt_stEst_mtxFbk, "delay": res})
    nest.Connect(stEst.pops_n,mc.fbk_p, "one_to_one", {"weight": -wgt_stEst_mtxFbk, "delay": res})
    nest.Connect(stEst.pops_n,mc.fbk_n, "one_to_one", {"weight": -wgt_stEst_mtxFbk, "delay": res})

# BRAIN STEM
brain_stem_new_p=[]
brain_stem_new_n=[]


for j in range(njt):
    # Positive neurons
    brain_stem_new_p = nest.Create ("basic_neuron_nestml", N)
    nest.SetStatus(brain_stem_new_p, {"kp": pops_params["brain_stem"]["kp"], "pos": True, "buffer_size": pops_params["brain_stem"]["buffer_size"], "base_rate": pops_params["brain_stem"]["base_rate"], "simulation_steps": len(time_vect)})
  
    # Negative neurons
    brain_stem_new_n = nest.Create ("basic_neuron_nestml", N)
    nest.SetStatus(brain_stem_new_n, {"kp": pops_params["brain_stem"]["kp"], "pos": False, "buffer_size": pops_params["brain_stem"]["buffer_size"], "base_rate": pops_params["brain_stem"]["base_rate"], "simulation_steps": len(time_vect)})
    


for j in range(njt):
    nest.Connect(mc.out_p,brain_stem_new_p, "all_to_all", {"weight": conn_params["mc_out_brain_stem"]["weight"], "delay": conn_params["mc_out_brain_stem"]["delay"]})
    #nest.Connect(mc.out_p,brain_stem_new_p, "all_to_all", {"weight": 100, "delay": conn_params["mc_out_brain_stem"]["delay"]})
    # nest.Connect(stEst.pops_p[j].pop,mc.fbk_n[j].pop, "one_to_one", {"weight": wgt_stEst_mtxFbk, "delay": res})
    # nest.Connect(stEst.pops_n[j].pop,mc.fbk_p[j].pop, "one_to_one", {"weight": -wgt_stEst_mtxFbk, "delay": res})
    nest.Connect(mc.out_n,brain_stem_new_n, "all_to_all", {"weight": -conn_params["mc_out_brain_stem"]["weight"], "delay": conn_params["mc_out_brain_stem"]["delay"]})

'''
# feedback from sensory
feedback_inv_p = nest.Create("diff_neuron", N)
nest.SetStatus(feedback_inv_p, {"kp": pops_params["feedback_inv"]["kp"], "pos": True, "buffer_size": pops_params["feedback_inv"]["buffer_size"], "base_rate": pops_params["feedback_inv"]["base_rate"]})
feedback_inv_n = nest.Create("diff_neuron", N)
nest.SetStatus(feedback_inv_n, {"kp": pops_params["feedback_inv"]["kp"], "pos": False, "buffer_size": pops_params["feedback_inv"]["buffer_size"], "base_rate": pops_params["feedback_inv"]["base_rate"]})
'''


###################### Extra Spikedetectors ######################

spikedetector_fbk_pos = nest.Create("spike_recorder", params={"label": "Feedback pos"})
spikedetector_fbk_neg = nest.Create("spike_recorder", params={"label": "Feedback neg"})


spikedetector_stEst_pos = nest.Create("spike_recorder", params={"label": "State estimator pos"})
spikedetector_stEst_neg = nest.Create("spike_recorder", params={"label": "State estimator neg"})

spikedetector_planner_pos = nest.Create("spike_recorder", params={"label": "Planner pos"})
spikedetector_planner_neg = nest.Create("spike_recorder", params={"label": "Planner neg"})

spikedetector_fbk_smoothed_pos = nest.Create("spike_recorder", params={"label": "Feedback smoothed pos"})
spikedetector_fbk_smoothed_neg = nest.Create("spike_recorder", params={"label": "Feedback smoothed neg"})

spikedetector_brain_stem_pos = nest.Create("spike_recorder", params={"label": "Brain stem pos"})
spikedetector_brain_stem_neg = nest.Create("spike_recorder", params={"label": "Brain stem neg"})

nest.Connect(brain_stem_new_p, spikedetector_brain_stem_pos)
nest.Connect(brain_stem_new_n, spikedetector_brain_stem_neg)


nest.Connect(planner.pops_p, spikedetector_planner_pos)
nest.Connect(planner.pops_n, spikedetector_planner_neg)


populations = {
    'sn_p': sn_p,
    'sn_n': sn_n,
    'spikedetector_brain_stem_pos': spikedetector_brain_stem_pos,
    'spikedetector_brain_stem_neg': spikedetector_brain_stem_neg
}
