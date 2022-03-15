import nest
import numpy as np


def connect_cereb_f(cerebellum_forw, prediction_p, prediction_n, error_p, error_n,  motor_commands_p, motor_commands_n ,n_forw ,conn_params):
    
    nest.Connect(cerebellum_forw.N_DCNp, prediction_p, 'all_to_all', syn_spec={"weight": conn_params["dcn_forw_prediction"]["weight"], "delay": conn_params["dcn_forw_prediction"]["delay"]})
    nest.Connect(cerebellum_forw.N_DCNn, prediction_p, 'all_to_all', syn_spec={"weight": -conn_params["dcn_forw_prediction"]["weight"], "delay": conn_params["dcn_forw_prediction"]["delay"]})
    nest.Connect(cerebellum_forw.N_DCNp, prediction_n, 'all_to_all', syn_spec={"weight": conn_params["dcn_forw_prediction"]["weight"], "delay": conn_params["dcn_forw_prediction"]["delay"]})
    nest.Connect(cerebellum_forw.N_DCNn, prediction_n, 'all_to_all', syn_spec={"weight": -conn_params["dcn_forw_prediction"]["weight"], "delay": conn_params["dcn_forw_prediction"]["delay"]})


    nest.Connect(motor_commands_p, cerebellum_forw.Nest_Mf[-n_forw:], {'rule': 'one_to_one'}) 
    nest.Connect(motor_commands_n, cerebellum_forw.Nest_Mf[0:n_forw], {'rule': 'one_to_one'})#TODO add weight


    # Construct the error signal for both positive and negative neurons
    nest.Connect(cerebellum_forw.N_DCNp, error_p, {'rule': 'all_to_all'}, syn_spec={"weight":conn_params["dcn_f_error"]["weight"]})
    nest.Connect(cerebellum_forw.N_DCNn, error_p, {'rule': 'all_to_all'}, syn_spec={"weight":-conn_params["dcn_f_error"]["weight"]})

    nest.Connect(cerebellum_forw.N_DCNp, error_n, {'rule': 'all_to_all'}, syn_spec={"weight":-conn_params["dcn_f_error"]["weight"]})
    nest.Connect(cerebellum_forw.N_DCNn, error_n, {'rule': 'all_to_all'}, syn_spec={"weight":conn_params["dcn_f_error"]["weight"]})


    # Connect error neurons toward IO neurons
    nest.Connect(error_p, cerebellum_forw.N_IOp,{'rule': 'all_to_all'}, conn_params["error_io_f"])
    nest.Connect(error_n, cerebellum_forw.N_IOn,{'rule': 'all_to_all'}, conn_params["error_io_f"])


def connect_state(stEst, sn_p, sn_n, njt, fbk_smoothed_p, fbk_smoothed_n, prediction_p, prediction_n, conn_params, cereb_controlled_joint):

    for j in range(njt):
        
        if j == cereb_controlled_joint:
            
            nest.Connect(sn_p[j].pop, fbk_smoothed_p, "all_to_all", syn_spec={"weight": conn_params["sn_fbk_smoothed"]["weight"], "delay": conn_params["sn_fbk_smoothed"]["delay"]})
            print(conn_params["sn_fbk_smoothed"]["weight"])
            nest.Connect(sn_n[j].pop, fbk_smoothed_n, "all_to_all", syn_spec={"weight": -conn_params["sn_fbk_smoothed"]["weight"], "delay": conn_params["sn_fbk_smoothed"]["delay"]})

            # Positive neurons
            nest.Connect(prediction_p, stEst.pops_p[j].pop, "all_to_all", syn_spec=conn_params["pred_state"])
            nest.Connect(fbk_smoothed_p, stEst.pops_p[j].pop, "all_to_all", syn_spec=conn_params["fbk_smoothed_state"])
            nest.SetStatus(stEst.pops_p[j].pop, {"num_first": float(N), "num_second": float(N)})

            # Negative neurons
            nest.Connect(prediction_n, stEst.pops_n[j].pop, "all_to_all", syn_spec=conn_params["pred_state"])
            nest.Connect(fbk_smoothed_n, stEst.pops_n[j].pop, "all_to_all", syn_spec=conn_params["fbk_smoothed_state"])
            nest.SetStatus(stEst.pops_n[j].pop, {"num_first": float(N), "num_second": float(N)})
        else:

            # Positive neurons
            nest.Connect(sn_p[j].pop, stEst.pops_p[j].pop, "all_to_all", syn_spec=conn_params["sn_state"])
            nest.SetStatus(stEst.pops_p[j].pop, {"num_second": float(N)})
            # Negative neurons
            nest.Connect(sn_n[j].pop, stEst.pops_n[j].pop, "all_to_all", syn_spec=conn_params["sn_state"])
            nest.SetStatus(stEst.pops_n[j].pop, {"num_second": float(N)})


def mc_connections(planner, mc, conn_params, stEst, motor_prediction_p, motor_prediction_n, cereb_controlled_joint, njt):

    #### Connection Planner - Motor Cortex feedback (excitatory)
    wgt_plnr_mtxFbk   = conn_params["planner_mc_fbk"]["weight"]

    # Delay between planner and motor cortex feedback.
    # It needs to compensate for the delay introduced by the state estimator
    #delay_plnr_mtxFbk = brain.stEst_param["buf_sz"] # USE THIS WITH REAL STATE ESTIMATOR
    delay_plnr_mtxFbk = conn_params["planner_mc_fbk"]["delay"]                         # USE THIS WITH "FAKE" STATE ESTIMATOR
    wgt_stEst_mtxFbk = conn_params["state_mc_fbk"]["weight"]
    for j in range(njt):
        planner.pops_p[j].connect( mc.fbk_p[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        planner.pops_p[j].connect( mc.fbk_n[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        planner.pops_n[j].connect( mc.fbk_p[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        planner.pops_n[j].connect( mc.fbk_n[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )

        # planner.pops_p[j].connect( mc.ffwd_p[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        # planner.pops_p[j].connect( mc.ffwd_n[j], rule='one_to_one', w= wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        # planner.pops_n[j].connect( mc.ffwd_p[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
        # planner.pops_n[j].connect( mc.ffwd_n[j], rule='one_to_one', w=-wgt_plnr_mtxFbk, d=delay_plnr_mtxFbk )
    # Connect state estimator (bayesian) to the Motor Cortex
    for j in range(njt):
        nest.Connect(stEst.pops_p[j].pop,mc.fbk_p[j].pop, "one_to_one", {"weight": wgt_stEst_mtxFbk, "delay": res})
        nest.Connect(stEst.pops_p[j].pop,mc.fbk_n[j].pop, "one_to_one", {"weight": wgt_stEst_mtxFbk, "delay": res})
        nest.Connect(stEst.pops_n[j].pop,mc.fbk_p[j].pop, "one_to_one", {"weight": -wgt_stEst_mtxFbk, "delay": res})
        nest.Connect(stEst.pops_n[j].pop,mc.fbk_n[j].pop, "one_to_one", {"weight": -wgt_stEst_mtxFbk, "delay": res})
    # connections to motor cortex # TODO ffwrd??
    nest.Connect(motor_prediction_p,mc.out_p[cereb_controlled_joint].pop, "one_to_one", conn_params["motor_pred_mc_out"])
    #nest.Connect(motor_prediction_p,mc.ffwd_n[j].pop, "all_to_all", {"weight": 0.1, "delay": res })
    #nest.Connect(motor_prediction_n,mc.ffwd_p[j].pop, "all_to_all", {"weight": 0.1, "delay": res })
    nest.Connect(motor_prediction_n,mc.out_n[cereb_controlled_joint].pop, "one_to_one", conn_params["motor_pred_mc_out"])


def modulators_connect(mc, planner, plan_to_inv_p, plan_to_inv_n, feedback_p, feedback_n, feedback_inv_p, feedback_inv_n, motor_commands_p, motor_commands_n, error_p, error_n, error_inv_p, error_inv_n, sn_p, sn_n, cereb_controlled_joint, conn_params):

    nest.Connect(mc.out_p[cereb_controlled_joint].pop, motor_commands_p, "all_to_all", syn_spec={"weight": conn_params["mc_out_motor_commands"]["weight"], "delay": conn_params["mc_out_motor_commands"]["delay"]})
    nest.Connect(mc.out_n[cereb_controlled_joint].pop, motor_commands_n, "all_to_all", syn_spec={"weight": -conn_params["mc_out_motor_commands"]["weight"], "delay": conn_params["mc_out_motor_commands"]["delay"]})

    nest.Connect(sn_p[cereb_controlled_joint].pop, feedback_p, 'all_to_all', syn_spec={"weight": conn_params["sn_feedback"]["weight"], "delay": conn_params["sn_feedback"]["delay"]})
    nest.Connect(sn_n[cereb_controlled_joint].pop, feedback_n, 'all_to_all', syn_spec={"weight": -conn_params["sn_feedback"]["weight"], "delay": conn_params["sn_feedback"]["delay"]})

    nest.Connect(feedback_p, error_p, 'all_to_all', syn_spec={"weight":conn_params["feedback_error"]["weight"]})
    nest.Connect(feedback_n, error_p, 'all_to_all', syn_spec={"weight":-conn_params["feedback_error"]["weight"]})
    nest.Connect(feedback_p, error_n, 'all_to_all', syn_spec={"weight":-conn_params["feedback_error"]["weight"]})
    nest.Connect(feedback_n, error_n, 'all_to_all', syn_spec={"weight":conn_params["feedback_error"]["weight"]})
    res = 0.1
    syn_exc = {"weight": 0.001, "delay": res} # 0.003
    syn_inh = {"weight": -0.001, "delay": res} #TODO set from json!!
    nest.Connect(planner.pops_p[cereb_controlled_joint].pop, plan_to_inv_p, "all_to_all", syn_spec=syn_exc)
    nest.Connect(planner.pops_n[cereb_controlled_joint].pop, plan_to_inv_n, "all_to_all", syn_spec=syn_inh)

    syn_exc = {"weight": 0.001, "delay": res}
    syn_inh = {"weight": -0.001, "delay": res}
    nest.Connect(sn_p[cereb_controlled_joint].pop, feedback_inv_p, 'all_to_all', syn_spec=conn_params["sn_feedback_inv"])
    nest.Connect(sn_n[cereb_controlled_joint].pop, feedback_inv_n, 'all_to_all', syn_spec=conn_params["sn_feedback_inv"])

    # Positive neurons
    nest.Connect(feedback_inv_p, error_inv_p, "all_to_all", syn_spec={"weight": conn_params["feedback_inv_error_inv"]["weight"], "delay": conn_params["feedback_inv_error_inv"]["delay"]})
    nest.Connect(feedback_inv_p, error_inv_n, "all_to_all", syn_spec={"weight": conn_params["feedback_inv_error_inv"]["weight"], "delay": conn_params["feedback_inv_error_inv"]["delay"]})

    # Negative neurons
    nest.Connect(feedback_inv_n, error_inv_n, "all_to_all", syn_spec={"weight": -conn_params["feedback_inv_error_inv"]["weight"], "delay": conn_params["feedback_inv_error_inv"]["delay"]})
    nest.Connect(feedback_inv_n, error_inv_p, "all_to_all", syn_spec={"weight": -conn_params["feedback_inv_error_inv"]["weight"], "delay": conn_params["feedback_inv_error_inv"]["delay"]})

    nest.Connect(plan_to_inv_p, error_inv_p, {'rule': 'all_to_all'}, syn_spec={"weight": conn_params["plan_to_inv_error_inv"]["weight"], "delay": conn_params["plan_to_inv_error_inv"]["delay"]})
    nest.Connect(plan_to_inv_n, error_inv_p, {'rule': 'all_to_all'}, syn_spec={"weight": -conn_params["plan_to_inv_error_inv"]["weight"], "delay": conn_params["plan_to_inv_error_inv"]["delay"]})


def cereb_i_connect(cerebellum, motor_prediction_p, motor_prediction_n, error_inv_p, error_inv_n, plan_to_inv_p, plan_to_inv_n, conn_params):

    nest.Connect(plan_to_inv_p, cerebellum.Nest_Mf[-n:], {'rule': 'one_to_one'}) #TODO weight
    nest.Connect(plan_to_inv_n, cerebellum.Nest_Mf[0:n], {'rule': 'one_to_one'})

    #syn_exc = {"weight": 0.3, "delay": res}
    #syn_inh = {"weight": -0.3, "delay": res}
    nest.Connect(cerebellum.N_DCNp, motor_prediction_p, 'all_to_all', syn_spec={"weight": conn_params["dcn_i_motor_pred"]["weight"], "delay": conn_params["dcn_i_motor_pred"]["delay"]})
    nest.Connect(cerebellum.N_DCNn, motor_prediction_p, 'all_to_all', syn_spec={"weight": -conn_params["dcn_i_motor_pred"]["weight"], "delay": conn_params["dcn_i_motor_pred"]["delay"]})
    nest.Connect(cerebellum.N_DCNp, motor_prediction_n, 'all_to_all', syn_spec={"weight": conn_params["dcn_i_motor_pred"]["weight"], "delay": conn_params["dcn_i_motor_pred"]["delay"]})
    nest.Connect(cerebellum.N_DCNn, motor_prediction_n, 'all_to_all', syn_spec={"weight": -conn_params["dcn_i_motor_pred"]["weight"], "delay": conn_params["dcn_i_motor_pred"]["delay"]})

    nest.Connect(error_inv_p, cerebellum.N_IOp,{'rule': 'all_to_all'}, conn_params["error_inv_io_i"])
    nest.Connect(error_inv_n, cerebellum.N_IOn,{'rule': 'all_to_all'}, conn_params["error_inv_io_i"])

