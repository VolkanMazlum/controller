import nest
import numpy as np
import json
from validation_synapse_sinexp import plot_synaptic_weight, reformat_weights, plot_synaptic_matrix, correct_spike_times
import sys



def test_null_parameters_sinexp():
    TEST_NUMBER = 8
    f = open('tests_plasticity/eglif_params.json')
    params = json.load(f)
    f.close()

    IO_params = params["EGLIF_model"]["IO"]
    GR_params = params["EGLIF_model"]["Granule"]
    PC_params = params["EGLIF_model"]["Purkinje"]

    nest.ResetKernel()
    nest.Install("custom_stdp_module")
    nest.SetKernelStatus({"resolution": 0.1})

    N = 3
    M = 5
    IO = nest.Create("eglif_io_nestml", N) # IO
    nest.SetStatus(IO, IO_params)
    PC = nest.Create("eglif_pc_nestml", N) # PC
    nest.SetStatus(PC, PC_params)
    Gr = nest.Create("eglif_cond_alpha_multisyn", M) # Gr
    #nest.SetStatus(Gr, GR_params) # If used, Gr does not spike


    # Decreasing the threshold of GrCs to increase their firing rate
    nest.SetStatus(Gr, {"V_th": -70.0})

    # Connections
    nest.Connect(IO, PC, "one_to_one", syn_spec={"receptor_type": 5})


    wr = nest.Create("weight_recorder")
    nest.CopyModel("stdp_synapse_sinexp", "my_stdp_synapse_rec", {"weight_recorder": wr, "Aplus": 0.0, "t_0":150, "Aminus": 0.0})
    conn_spec_dict = {'rule': 'fixed_indegree', 'indegree': 4}
    syn_spec_dict = {'synapse_model': "my_stdp_synapse_rec", "receptor_type": 1}
    nest.Connect(Gr, PC, conn_spec = conn_spec_dict, syn_spec = syn_spec_dict)

    # Devices
    cf_recorder = nest.Create("spike_recorder", {"record_to": "memory"})
    PC_recorder = nest.Create("spike_recorder",  {"record_to": "memory"})
    Gr_recorder = nest.Create("spike_recorder",  {"record_to": "memory"})

    nest.Connect(IO, cf_recorder)
    nest.Connect(PC, PC_recorder)
    nest.Connect(Gr, Gr_recorder)


    pf_pc_conns = nest.GetConnections(synapse_model="my_stdp_synapse_rec")

    weights = []
    for i in range(1000):
        nest.Simulate(1.0)
        weights.append(nest.GetStatus(pf_pc_conns, "weight"))

    weight_matrix = reformat_weights(weights)

    corrected_cf_tms, cf_evs, corrected_PC_tms, PC_evs, PC_complex_tms, PC_complex_evs, corrected_gr_spikes, Gr_evs = correct_spike_times(cf_recorder, PC_recorder, Gr_recorder)

    data = nest.GetStatus(wr)[0]

    plot_synaptic_weight(pf_pc_conns, data, IO, PC, Gr, corrected_cf_tms, cf_evs, corrected_gr_spikes, Gr_evs, corrected_PC_tms, PC_evs, TEST_NUMBER)

    assert (np.all(weight_matrix == 0.14 ))