import nest
import numpy as np
import json
from validation_synapse_alpha import plot_synaptic_weight, reformat_weights, plot_synaptic_matrix, correct_spike_times, plot_LTP
import sys



f = open('eglif_params.json')
params = json.load(f)
f.close()

IO_params = params["EGLIF_model"]["IO"]
GR_params = params["EGLIF_model"]["Granule"]
MLI_params = params["EGLIF_model"]["Stellate"]

N = 3
M = 5

Aminus_value = 0.0
Aplus_values = [ 0.0001 ,  0.0005,  0.0006 ]

for trial in range(N):
    TEST_NUMBER = 6 + trial/10
    nest.ResetKernel()
    nest.Install("custom_stdp_module")
    nest.SetKernelStatus({"resolution": 0.1})

    # Create the network and tune parameters
    IO = nest.Create("eglif_io_nestml", N) # IO
    nest.SetStatus(IO, IO_params)
    MLI = nest.Create("eglif_mli", N) # PC
    nest.SetStatus(MLI, MLI_params)
    Gr = nest.Create("eglif_cond_alpha_multisyn", M) # Gr
    #nest.SetStatus(Gr, GR_params) # If used, Gr does not spike

    # Connections
    nest.Connect(IO, MLI, "one_to_one", syn_spec={"receptor_type": 3})

    
    wr = nest.Create("weight_recorder")
    nest.CopyModel("stdp_synapse_alpha", "my_stdp_synapse_rec", {"weight_recorder": wr,  "Aplus": Aplus_values[trial], "Aminus": -0.000})
    
    conn_spec_dict = {'rule': 'fixed_indegree', 'indegree': 4}
    syn_spec_dict = {'synapse_model': "my_stdp_synapse_rec", "receptor_type": 1}
    nest.Connect(Gr, MLI, conn_spec = conn_spec_dict, syn_spec = syn_spec_dict)

    pf_pc_conns = nest.GetConnections( synapse_model="my_stdp_synapse_rec")
    IO_pc_conns = nest.GetConnections(source = IO, target = MLI)
    
    # Devices   
    cf_recorder = nest.Create("spike_recorder", {"record_to": "memory"})
    MLI_recorder = nest.Create("spike_recorder",  {"record_to": "memory"})
    Gr_recorder = nest.Create("spike_recorder",  {"record_to": "memory"})

    nest.Connect(IO, cf_recorder)
    nest.Connect(MLI, MLI_recorder)
    nest.Connect(Gr, Gr_recorder)

    # SIMULATION
    weights = []
    for i in range(1000):
        nest.Simulate(1.0)
        weights.append(nest.GetStatus(pf_pc_conns, "weight"))

    weight_matrix = reformat_weights(weights)
    print(weight_matrix)
    corrected_cf_tms, cf_evs, corrected_MLI_tms, MLI_evs, MLI_complex_tms, MLI_complex_evs, corrected_gr_spikes, Gr_evs = correct_spike_times(cf_recorder, MLI_recorder, Gr_recorder)
    data = nest.GetStatus(wr)[0]
    
    
    plot_synaptic_weight(pf_pc_conns, data, IO, MLI, Gr, corrected_cf_tms, cf_evs, corrected_gr_spikes, Gr_evs, corrected_MLI_tms, MLI_evs, TEST_NUMBER) 
    
    count_errors = []
    
    # Assert that if there is LTP, it is of Aplus
    for n, conn in enumerate(pf_pc_conns):
        
        diff_weights = np.round(np.diff(weight_matrix[n, :]), 3)

        assert np.all(diff_weights >= 0)
        for timestep, diff in enumerate(diff_weights):
            if diff > 0 and diff != Aplus_values[trial]:
                count_errors.append(diff)
                #print(count_errors)
                

