import nest
from population_view import PopView

def spine_pop_init(params, N, njt, time_vect):

    delay_fbk          = params["modules"]["spine"]["fbk_delay"]
    wgt_sensNeur_spine = params["modules"]["spine"]["wgt_sensNeur_spine"]

    #### Sensory feedback (Parrot neurons on Sensory neurons)
    sn_p=[]
    sn_n=[]

    for j in range(njt):
        # Positive neurons
        tmp_p = nest.Create ("parrot_neuron", N)
        sn_p.append( PopView(tmp_p, time_vect, to_file=True, label='sens_fbk_'+str(j)+'_p') )
        # Negative neurons
        tmp_n = nest.Create ("parrot_neuron", N)
        sn_n.append( PopView(tmp_n, time_vect, to_file=True, label='sens_fbk_'+str(j)+'_n') )

    return sn_p, sn_n

def prediction_pop_init(pops_params, N):

    prediction_p = nest.Create("diff_neuron", N)
    nest.SetStatus(prediction_p, {"kp": pops_params["prediction"]["kp"], "pos": True, "buffer_size": pops_params["prediction"]["buffer_size"], "base_rate": pops_params["prediction"]["base_rate"]}) #5.5
    prediction_n = nest.Create("diff_neuron", N)
    nest.SetStatus(prediction_n, {"kp": pops_params["prediction"]["kp"], "pos": False, "buffer_size": pops_params["prediction"]["buffer_size"], "base_rate": pops_params["prediction"]["base_rate"]}) #5.5
    
    return prediction_p, prediction_n

def fbk_smoothed_pop_init(pops_params, N):

    fbk_smoothed_p = nest.Create("diff_neuron", N)
    nest.SetStatus(fbk_smoothed_p, {"kp": pops_params["fbk_smoothed"]["kp"], "pos": True, "buffer_size": pops_params["fbk_smoothed"]["buffer_size"], "base_rate": pops_params["fbk_smoothed"]["base_rate"]})
    fbk_smoothed_n = nest.Create("diff_neuron", N)
    nest.SetStatus(fbk_smoothed_n, {"kp": pops_params["fbk_smoothed"]["kp"], "pos": False, "buffer_size": pops_params["fbk_smoothed"]["buffer_size"], "base_rate": pops_params["fbk_smoothed"]["base_rate"]})
    return fbk_smoothed_p, fbk_smoothed_n


def motor_commands_pop_init(pops_params,n_forw):
    motor_commands_p = nest.Create("diff_neuron", n_forw)
    nest.SetStatus(motor_commands_p, {"kp": pops_params["motor_commands"]["kp"], "pos": True, "buffer_size": pops_params["motor_commands"]["buffer_size"], "base_rate": pops_params["motor_commands"]["base_rate"]})
    motor_commands_n = nest.Create("diff_neuron", n_forw)
    nest.SetStatus(motor_commands_n, {"kp": pops_params["motor_commands"]["kp"], "pos": False, "buffer_size": pops_params["motor_commands"]["buffer_size"], "base_rate": pops_params["motor_commands"]["base_rate"]})
    return motor_commands_p, motor_commands_n

def feedback_pop_init(pops_params, N):
    feedback_p = nest.Create("diff_neuron", N)
    nest.SetStatus(feedback_p, {"kp": pops_params["feedback"]["kp"], "pos": True, "buffer_size": pops_params["feedback"]["buffer_size"], "base_rate": pops_params["feedback"]["base_rate"]})
    feedback_n = nest.Create("diff_neuron", N)
    nest.SetStatus(feedback_n, {"kp": pops_params["feedback"]["kp"], "pos": False, "buffer_size": pops_params["feedback"]["buffer_size"], "base_rate": pops_params["feedback"]["base_rate"]})

    return feedback_p,feedback_n

def error_f_pop_init(pops_params, N):
    # Error signal toward IO neurons ############
    ''
    # Positive subpopulation
    error_p = nest.Create("diff_neuron", N)
    nest.SetStatus(error_p, {"kp": pops_params["error"]["kp"], "pos": True, "buffer_size":pops_params["error"]["buffer_size"], "base_rate": pops_params["error"]["base_rate"]})
    # Negative subpopulation
    error_n = nest.Create("diff_neuron", N)
    nest.SetStatus(error_n, {"kp": pops_params["error"]["kp"], "pos": False, "buffer_size":pops_params["error"]["buffer_size"], "base_rate": pops_params["error"]["base_rate"]})
    return error_p, error_n

def error_i_pop_init(pops_params, N):
    error_inv_p = nest.Create("diff_neuron", N)
    nest.SetStatus(error_inv_p, {"kp": pops_params["error_i"]["kp"], "pos": True, "buffer_size":pops_params["error_i"]["buffer_size"], "base_rate": pops_params["error_i"]["base_rate"]})
    # Negative subpopulation
    error_inv_n = nest.Create("diff_neuron", N)
    nest.SetStatus(error_inv_n, {"kp": pops_params["error_i"]["kp"], "pos": False, "buffer_size":pops_params["error_i"]["buffer_size"], "base_rate": pops_params["error_i"]["base_rate"]})
    return error_inv_p, error_inv_n

def plan_to_inv_pop_init(pops_params, n):

    # Input to inverse neurons
    plan_to_inv_p = nest.Create("diff_neuron", n)
    nest.SetStatus(plan_to_inv_p, {"kp": pops_params["plan_to_inv"]["kp"], "pos": True, "buffer_size": pops_params["plan_to_inv"]["buffer_size"],  "base_rate": pops_params["plan_to_inv"]["base_rate"]})
    plan_to_inv_n = nest.Create("diff_neuron", n)
    nest.SetStatus(plan_to_inv_n, {"kp": pops_params["plan_to_inv"]["kp"], "pos": False, "buffer_size": pops_params["plan_to_inv"]["buffer_size"], "base_rate": pops_params["plan_to_inv"]["base_rate"]})
    return plan_to_inv_p, plan_to_inv_n


def motor_pred_pop_init(pops_params,N):
    motor_prediction_p = nest.Create("diff_neuron", N)
    nest.SetStatus(motor_prediction_p, {"kp": pops_params["motor_pred"]["kp"], "pos": True, "buffer_size": pops_params["motor_pred"]["buffer_size"], "base_rate": pops_params["motor_pred"]["base_rate"]})
    motor_prediction_n = nest.Create("diff_neuron", N)
    nest.SetStatus(motor_prediction_n, {"kp": pops_params["motor_pred"]["kp"], "pos": False, "buffer_size": pops_params["motor_pred"]["buffer_size"], "base_rate": pops_params["motor_pred"]["base_rate"]})
    return motor_prediction_p, motor_prediction_n

def feedback_i_pop_init(pops_params, N):
    # feedback from sensory
    feedback_inv_p = nest.Create("diff_neuron", N)
    nest.SetStatus(feedback_inv_p, {"kp": pops_params["feedback_inv"]["kp"], "pos": True, "buffer_size": pops_params["feedback_inv"]["buffer_size"], "base_rate": pops_params["feedback_inv"]["base_rate"]})
    feedback_inv_n = nest.Create("diff_neuron", N)
    nest.SetStatus(feedback_inv_n, {"kp": pops_params["feedback_inv"]["kp"], "pos": False, "buffer_size": pops_params["feedback_inv"]["buffer_size"], "base_rate": pops_params["feedback_inv"]["base_rate"]})
    
    return feedback_inv_p, feedback_inv_n



# def spikes_detector_init():
#     spikedetector_fbk_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback pos"})
#     spikedetector_fbk_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback neg"})
#     spikedetector_fbk_cereb_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback cerebellum pos"})
#     spikedetector_fbk_cereb_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback cerebellum neg"})
#     spikedetector_io_forw_input_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Input inferior Olive Forw pos"})
#     spikedetector_io_forw_input_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Input inferior Olive Forw neg"})

#     spikedetector_io_inv_input_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Input inferior Olive Inv pos"})
#     spikedetector_io_inv_input_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Input inferior Olive Inv neg"})

#     spikedetector_stEst_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "State estimator pos"})
#     spikedetector_stEst_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "State estimator neg"})
#     spikedetector_planner_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Planner pos"})
#     spikedetector_planner_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Planner neg"})
#     spikedetector_pred_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Cereb pred pos"})
#     spikedetector_pred_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Cereb pred neg"})
#     spikedetector_motor_pred_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Cereb motor pred pos"})
#     spikedetector_motor_pred_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Cereb motor pred neg"})
#     spikedetector_stEst_max_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "State estimator Max pos"})
#     spikedetector_stEst_max_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "State estimator Max neg"})
#     spikedetector_fbk_smoothed_pos = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback smoothed pos"})
#     spikedetector_fbk_smoothed_neg = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "Feedback smoothed neg"})
#     return spikedetector_fbk_pos,spikedetector_fbk_neg,spikedetector_fbk_cereb_pos