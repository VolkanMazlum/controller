"""Motor cortex class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt
import nest

class MotorCortex:
    ############## Constructor (plant value is just for testing) ##############
    def __init__(self, numNeurons, numJoints, time_vect, mtCmds, **kwargs):

        ### Initialize neural network
        # General parameters of neurons
        param_neurons = {
            "ffwd_base_rate": 0.0, # Feedforward neurons
            "ffwd_kp": 1.0,
            "fbk_base_rate": 0.0,  # Feedback neurons
            "fbk_kp": 1.0,
            "out_base_rate": 0.0,  # Summation neurons
            "out_kp":1.0,
            "wgt_ffwd_out": 1.0,   # Connection weight from ffwd to output neurons (must be positive)
            "wgt_fbk_out": 1.0,    # Connection weight from fbk to output neurons (must be positive)
            "buf_sz": 10.0         # Size of the buffer to compute spike rate in basic_neurons (ms)
            }
        param_neurons.update(kwargs)
        # Motor commands
        self.motorCommands = mtCmds
        self.numJoints = numJoints
        # Time vector
        self.time_vect = time_vect
        # Initialize
        self.init_neurons(numNeurons, param_neurons)

    ############################ Motor commands ############################
    '''
        # Include pause into the motor commands
        res = nest.GetKernelStatus({"resolution"})[0]
        time_bins = commands.shape[0] + int(self.pause_len/res)
        commands_pause = np.zeros((time_bins, commands.shape[1])) 
        for i in range(nj):
            commands_pause[:,i] = AddPause(commands[:,i], self.pause_len, res)

        # save joint trajectories into files
        # NOTE: THIS OVERWRITES EXISTING TRAJECTORIES
        for i in range(nj):
            cmd_file = self.pathData + "joint_cmd_"+str(i)+".dat"
            a_file = open(cmd_file, "w")
            np.savetxt( a_file, commands_pause[:,i] )
            a_file.close()
    '''

    ######################## Initialize neural network #########################
    def init_neurons(self, numNeurons, params):

        par_ffwd = {
            "base_rate": params['ffwd_base_rate'],
            "kp": params['ffwd_kp']
        }

        par_fbk = {
            "base_rate": params['fbk_base_rate'],
            "kp": params['fbk_kp']
        }

        par_out = {
            "base_rate": params['out_base_rate'],
            "kp": params['out_kp']
        }

        buf_sz = params['buf_sz']

        res = self.time_vect[1]-self.time_vect[0]

        # Create populations
        for i in range(self.numJoints):

            ############ FEEDFORWARD POPULATION ############
            # Positive and negative populations for each joint
            # Positive population (joint i)
            self.ffwd_p = nest.Create("tracking_neuron_nestml", n=numNeurons, params=par_ffwd)
            nest.SetStatus(self.ffwd_p, {"pos": True, "traj": self.motorCommands, "simulation_steps": len(self.motorCommands)})
            
            # Negative population (joint i)
            self.ffwd_n = nest.Create("tracking_neuron_nestml", n=numNeurons, params=par_ffwd)
            nest.SetStatus(self.ffwd_n, {"pos": False, "traj": self.motorCommands, "simulation_steps": len(self.motorCommands)})

            
            ############ FEEDBACK POPULATION ############
            # Positive and negative populations for each joint

            # Positive population (joint i)
            self.fbk_p = nest.Create("diff_neuron_nestml", n=numNeurons, params=par_fbk)
            nest.SetStatus(self.fbk_p, {"pos": True, "buffer_size": buf_sz, "simulation_steps": len(self.time_vect)})

            # Negative population (joint i)
            self.fbk_n = nest.Create("diff_neuron_nestml", n=numNeurons, params=par_fbk)
            nest.SetStatus(self.fbk_n, {"pos": False, "buffer_size": buf_sz, "simulation_steps": len(self.time_vect)})

            ############ OUTPUT POPULATION ############
            # Positive and negative populations for each joint.
            # Here I could probably just use a neuron that passes the spikes it receives  from
            # the connected neurons (excitatory), rather tahan computing the frequency in a buffer
            # and draw from Poisson (i.e. basic_neuron).

            # Positive population (joint i)
            self.out_p = nest.Create("basic_neuron_nestml", n=numNeurons, params=par_out)
            nest.SetStatus(self.out_p, {"pos": True, "buffer_size": buf_sz, "simulation_steps": len(self.time_vect)})
            
            # Negative population (joint i)
            self.out_n = nest.Create("basic_neuron_nestml", n=numNeurons, params=par_out)
            nest.SetStatus(self.out_n, {"pos": False, "buffer_size": buf_sz, "simulation_steps": len(self.time_vect)})
            
            ###### CONNECT FFWD AND FBK POULATIONS TO OUT POPULATION ######
            # Populations of each joint are connected together according to connection
            # rules and network topology. There is no connections across joints
            nest.Connect(self.ffwd_p, self.out_p, 'one_to_one', {"weight": params['wgt_ffwd_out'], "delay": res})
            nest.Connect(self.ffwd_p, self.out_n, 'one_to_one', {"weight": params['wgt_ffwd_out'], "delay": res})
            nest.Connect(self.ffwd_n, self.out_p, 'one_to_one', {"weight": params['wgt_ffwd_out'], "delay": res})
            nest.Connect(self.ffwd_n, self.out_n, 'one_to_one', {"weight": params['wgt_ffwd_out'], "delay": res})
            
            nest.Connect(self.fbk_p, self.out_p, 'one_to_one', {"weight": params['wgt_fbk_out'], "delay": res})
            nest.Connect(self.fbk_p, self.out_n, 'one_to_one', {"weight": params['wgt_fbk_out'], "delay": res})
            nest.Connect(self.fbk_n, self.out_p, 'one_to_one', {"weight": params['wgt_fbk_out'], "delay": res})
            nest.Connect(self.fbk_n, self.out_n, 'one_to_one', {"weight": params['wgt_fbk_out'], "delay": res})
            
