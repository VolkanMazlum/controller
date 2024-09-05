from nrp_core import *
from nrp_core.data.nrp_protobuf import DumpStringDataPack
from nrp_core.data.nrp_protobuf import DumpArrayFloatDataPack
import json
import numpy as np

pathData = os.getcwd()+'/dependencies/data/spike_recorders.json'
@EngineDataPack(keyword='planner_p', id=DataPackIdentifier('planner_p', 'nest'))
@EngineDataPack(keyword='planner_n', id=DataPackIdentifier('planner_n', 'nest'))
@EngineDataPack(keyword='ffwd_p', id=DataPackIdentifier('ffwd_p', 'nest'))
@EngineDataPack(keyword='ffwd_n', id=DataPackIdentifier('ffwd_n', 'nest'))
@EngineDataPack(keyword='fbk_p', id=DataPackIdentifier('fbk_p', 'nest'))
@EngineDataPack(keyword='fbk_n', id=DataPackIdentifier('fbk_n', 'nest'))
@EngineDataPack(keyword='out_p', id=DataPackIdentifier('out_p', 'nest'))
@EngineDataPack(keyword='out_n', id=DataPackIdentifier('out_n', 'nest'))
@EngineDataPack(keyword='stEst_p', id=DataPackIdentifier('stEst_p', 'nest'))
@EngineDataPack(keyword='stEst_n', id=DataPackIdentifier('stEst_n', 'nest'))
@EngineDataPack(keyword='brainstem_n', id=DataPackIdentifier('spikedetector_brain_stem_neg', 'nest'))
@EngineDataPack(keyword='brainstem_p', id=DataPackIdentifier('spikedetector_brain_stem_pos', 'nest'))
@EngineDataPack(keyword='sn_parrot_p', id=DataPackIdentifier('sn_parrot_p', 'nest'))
@EngineDataPack(keyword='sn_parrot_n', id=DataPackIdentifier('sn_parrot_n', 'nest'))
@EngineDataPack(keyword='pred_p', id=DataPackIdentifier('pred_p', 'nest'))
@EngineDataPack(keyword='pred_n', id=DataPackIdentifier('pred_n', 'nest'))
@EngineDataPack(keyword='sn_p', id=DataPackIdentifier('sn_p', 'nest'))
@EngineDataPack(keyword='sn_n', id=DataPackIdentifier('sn_n', 'nest'))
@TransceiverFunction("datatransfer_engine")
def data_transfer(planner_p, planner_n, ffwd_p, ffwd_n, fbk_p, fbk_n, out_p, out_n, stEst_p, stEst_n, brainstem_n, brainstem_p, sn_parrot_p, sn_parrot_n, pred_p, pred_n, sn_p, sn_n):
    
    global pathData
    #global spike_recordings
    planner = {
        "pos": {
            "times": planner_p.data[0]['events']['times'],  
            "senders": planner_p.data[0]['events']['senders']   
        },
        "neg": {
            "times": planner_n.data[0]['events']['times'],    
            "senders": planner_n.data[0]['events']['senders']  
        }
    }

    motor_cortex = {
        "ffwd": {
            "pos": {
                "times": ffwd_p.data[0]['events']['times'],  
                "senders": ffwd_p.data[0]['events']['senders']
            },
            "neg": {
                "times": ffwd_n.data[0]['events']['times'],  
                "senders": ffwd_n.data[0]['events']['senders']
            }
        },
        "fbk": {
            "pos": {
                "times": fbk_p.data[0]['events']['times'],  
                "senders": fbk_p.data[0]['events']['senders']
            },
            "neg": {
                "times": fbk_n.data[0]['events']['times'],  
                "senders": fbk_n.data[0]['events']['senders']
            }
        },
        "out": {
            "pos": {
                "times": out_p.data[0]['events']['times'],  
                "senders": out_p.data[0]['events']['senders']
            },
            "neg": {
                "times": out_n.data[0]['events']['times'],  
                "senders": out_n.data[0]['events']['senders']
            }
        }
    }

    state = {
        "pos": {
            "times": stEst_p.data[0]['events']['times'],
            "senders": stEst_p.data[0]['events']['senders']
        },
        "neg": {
            "times": stEst_n.data[0]['events']['times'],
            "senders": stEst_n.data[0]['events']['senders']
        }
    }

    brainstem = {
        "pos": {
            "times": brainstem_p.data[0]['events']['times'],
            "senders": brainstem_p.data[0]['events']['senders']
        },
        "neg": {
            "times": brainstem_n.data[0]['events']['times'],
            "senders": brainstem_n.data[0]['events']['senders']
        }
    }

    sensory = {
        "pos": {
            "times": sn_parrot_p.data[0]['events']['times'],
            "senders": sn_parrot_p.data[0]['events']['senders']
        },
        "neg": {
            "times": sn_parrot_n.data[0]['events']['times'],
            "senders": sn_parrot_n.data[0]['events']['senders']
        }
    }

    prediction = {
        "pos": {
            "times": pred_p.data[0]['events']['times'],
            "senders": pred_p.data[0]['events']['senders']
        },
        "neg": {
            "times": pred_n.data[0]['events']['times'],
            "senders": pred_n.data[0]['events']['senders']
        }
    }

    spike_recordings = {
        "planner": planner,
        "motor_cortex": motor_cortex,
        "state": state,
        "brainstem": brainstem,
        "sensory": sensory,
        "prediction": prediction
    }


    # Saving data to a JSON file
    with open(pathData, 'w') as file:
        json.dump(spike_recordings, file, indent=4)
    
    # Streaming formatted arrays
    # Brainstem
    bs_p_sds = DumpArrayFloatDataPack("bs_p_sds", "datatransfer_engine")
    #bs_p_sds.data.float_stream.extend(brainstem_p.data[0]['events']['senders'])
    '''
    bs_p_ts = DumpArrayFloatDataPack("bs_p_ts", "datatransfer_engine")
    bs_p_ts.data.float_stream.extend(brainstem_p.data[0]['events']['times'])
    
    bs_n_sds = DumpArrayFloatDataPack("bs_n_sds", "datatransfer_engine")
    bs_p_sds.data.float_stream.extend(brainstem_n.data[0]['events']['senders'])
    
    bs_n_ts = DumpArrayFloatDataPack("bs_n_ts", "datatransfer_engine")
    bs_n_ts.data.float_stream.extend(brainstem_n.data[0]['events']['times'])
    '''
    return [bs_p_sds]
