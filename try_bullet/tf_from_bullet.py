from nrp_core import *
from nrp_core.data.nrp_json import *

#car_speed = 5

base_rate = 0.0
kp = 1
res = 0.1



@EngineDataPack(keyword='positions', id=DataPackIdentifier('positions', 'bullet_simulator'))
@EngineDataPack(keyword='sn_p', id=DataPackIdentifier('sn_p', 'nest'))
@EngineDataPack(keyword='sn_n', id=DataPackIdentifier('sn_n', 'nest'))
#@PreprocessingFunction("bullet_simulator")
@TransceiverFunction("nest")
def from_bullet(positions, sn_p, sn_n):
    sn_p = JsonDataPack("sn_p", "nest")
    sn_n = JsonDataPack("sn_n", "nest")

    global base_rate
    global kp
    global res
    
    # Get data from bullet
    #signal = positions.data["hand"][-1]
    signal = positions.data["elbow"]
    rate = base_rate + kp * abs(signal)
    lmbd = rate * res
    
    if signal >= 0:
        sn_p.data['rate'] = lmbd
        sn_n.data['rate'] = 0.0
        #print("from_bullet: calculated lambda pos", lmbd)
    else:
    	sn_n.data['rate'] = lmbd
    	sn_p.data['rate'] = 0.0
    	#print("from_bullet: calculated lambda neg", lmbd)

    return [sn_p, sn_n]
