"""Motor cortex class"""

__authors__ = "Cristiano Alessandro"
__copyright__ = "Copyright 2021"
__credits__ = ["Cristiano Alessandro"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt
import nest
import trajectories as tj
from population_view import PopView
from util import AddPause

class MotorCortex:

    ############## Constructor (plant value is just for testing) ##############
    def __init__(self, numNeurons, time_vect, traj_joint, plant, pathData="./data/", precise=False, pause_len = 0.0, **kwargs):

        # Path where to save the data file
        self.pathData = pathData

        # If True, forward motor cortex generate precise motor commands
        # through inverse dynamics
        self.precise = precise

        # Time pause after task execution
        self.pause_len = pause_len

        ### Initialize analog signals
        self.init_analog(time_vect, traj_joint, plant)

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

        # Initialize
        self.init_neurons(numNeurons, param_neurons)


    ######################## Initialize analog signals ########################
    def init_analog(self, time_vect, traj_joint, plant):

        # Model of the plant to be controlled
        self.plant = plant
        self.numJoints = self.plant.numVariables()

        # Time vector
        self.time_vect = time_vect

        # Generate initial motor commands
        self.generateMotorCommands(traj_joint)


    ############## Set desired joint trajectory ##############
    def getJointTrajectory(self):
        return self.traj

    def setJointTrajectory(self, traj_joint):
        self.traj = traj_joint


    ############################ Motor commands ############################

    # Get motor commands
    def getMotorCommands(self):
        return self.motorCommands


    # General function to generate motor commands given joint trajectory
    def generateMotorCommands(self, trajectory):
        self.setJointTrajectory(trajectory)
        initPos = trajectory[0,:]                    # Initial position
        desPos  = trajectory[ len(trajectory)-1,: ]  # Desired target position
        if self.precise:
            self.motorCommands = self.generatePreciseMotorCommnads(trajectory)
        else:
            self.motorCommands = self.generateCoarseMotorCommnads(initPos, desPos, self.time_vect/1e3)

        # Save motor commands into files
        self.saveMotorCommands(self.motorCommands)


    # Precise motor commands using inverse dynamics
    def generatePreciseMotorCommnads(self, pos):
        dt  = (self.time_vect[1]-self.time_vect[0])/1e3
        vel = np.gradient(pos,dt,axis=0)
        acc = np.gradient(vel,dt,axis=0)
        mcmd = self.plant.inverseDyn( pos, vel, acc )
        return mcmd


    # Coarse motor commands using initial and desired position of each joint
    # (this assumes a reaching movement with acceleration and deceleration)
    def generateCoarseMotorCommnads(self, init_pos, des_pos, time_vector):

        if len(init_pos)!=self.numJoints | len(des_pos)!=self.numJoints:
            raise Exception("Number of joint is different from number of columns")

        if time_vector.size!=self.time_vect.size:
            raise Exception("Time vector is not well sized")

        # Last simulation time
        T_max = time_vector[ len(time_vector)-1 ]

        # Time and value of the minimum jerk curve
        ext_t, ext_val = tj.minJerk_ddt_minmax(init_pos, des_pos, time_vector)

        # Approximate with sin function
        tmp_ext = np.reshape( ext_val[0,:], (1,self.numJoints) ) # First extreme (positive)
        tmp_sin = np.sin( (2*np.pi*time_vector/T_max) )
        tmp_sin = np.reshape( tmp_sin,(tmp_sin.size,1) )

        # Motor commands: Inverse dynamics given approximated acceleration
        dt   = (self.time_vect[1]-self.time_vect[0])/1e3
        pos,pol  = tj.minimumJerk(init_pos, des_pos, time_vector)
        vel  = np.gradient(pos,dt,axis=0)
        acc  = tmp_ext*tmp_sin
        mcmd = self.plant.inverseDyn(pos, vel, acc)

        return mcmd


    # Save motor commands into files
    def saveMotorCommands(self, commands):

        nj = commands.shape[1]
        if nj!=self.numJoints:
            raise Exception("Number of joint is different from number of columns")

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

        self.ffwd_p = []
        self.ffwd_n = []
        self.fbk_p  = []
        self.fbk_n  = []
        self.out_p  = []
        self.out_n  = []

        res = self.time_vect[1]-self.time_vect[0]

        # Create populations
        for i in range(self.numJoints):

            ############ FEEDFORWARD POPULATION ############
            # Positive and negative populations for each joint

            # File where the trajectory is saved (from trajectory object)
            cmd_file = self.pathData + "joint_cmd_"+str(i)+".dat"
            #TODO: check if files exist

            # Positive population (joint i)
            tmp_pop_p = nest.Create("tracking_neuron_nestml", n=numNeurons, params=par_ffwd)
            nest.SetStatus(tmp_pop_p, {"pos": True, "traj": self.motorCommands.flatten().tolist(), "simulation_steps": len(self.motorCommands.flatten().tolist())})
            self.ffwd_p.append( PopView(tmp_pop_p,self.time_vect) )

            # Negative population (joint i)
            tmp_pop_n = nest.Create("tracking_neuron_nestml", n=numNeurons, params=par_ffwd)
            nest.SetStatus(tmp_pop_n, {"pos": False, "traj": self.motorCommands.flatten().tolist(), "simulation_steps": len(self.motorCommands.flatten().tolist())})
            self.ffwd_n.append( PopView(tmp_pop_n,self.time_vect) )

            '''
            ############ FEEDBACK POPULATION ############
            # Positive and negative populations for each joint

            # Positive population (joint i)
            #tmp_pop_p = nest.Create("basic_neuron", n=numNeurons, params=par_fbk)
            tmp_pop_p = nest.Create("diff_neuron", n=numNeurons, params=par_fbk)
            nest.SetStatus(tmp_pop_p, {"pos": True, "buffer_size": buf_sz})
            self.fbk_p.append( PopView(tmp_pop_p,self.time_vect) )

            # Negative population (joint i)
            #tmp_pop_n = nest.Create("basic_neuron", n=numNeurons, params=par_fbk)
            tmp_pop_n = nest.Create("diff_neuron", n=numNeurons, params=par_fbk)
            nest.SetStatus(tmp_pop_n, {"pos": False, "buffer_size": buf_sz})
            self.fbk_n.append( PopView(tmp_pop_n,self.time_vect) )

            ############ OUTPUT POPULATION ############
            # Positive and negative populations for each joint.
            # Here I could probably just use a neuron that passes the spikes it receives  from
            # the connected neurons (excitatory), rather tahan computing the frequency in a buffer
            # and draw from Poisson (i.e. basic_neuron).

            # Positive population (joint i)
            filename = self.pathData+"mc_out_p_"+str(i)
            filename = "mc_out_p_"+str(i)
            tmp_pop_p = nest.Create("basic_neuron", n=numNeurons, params=par_out)
            #tmp_pop_p = nest.Create("diff_neuron", n=numNeurons, params=par_out)
            nest.SetStatus(tmp_pop_p, {"pos": True, "buffer_size": buf_sz})
            #self.out_p.append( PopView(tmp_pop_p,self.time_vect) )
            self.out_p.append( PopView(tmp_pop_p,self.time_vect,to_file=True,label=filename) )

            # Negative population (joint i)
            filename = self.pathData+"mc_out_n_"+str(i)
            filename = "mc_out_n_"+str(i)
            tmp_pop_n = nest.Create("basic_neuron", n=numNeurons, params=par_out)
            #tmp_pop_n = nest.Create("diff_neuron", n=numNeurons, params=par_out)
            nest.SetStatus(tmp_pop_n, {"pos": False, "buffer_size": buf_sz})
            #self.out_n.append( PopView(tmp_pop_n,self.time_vect) )
            self.out_n.append( PopView(tmp_pop_n,self.time_vect,to_file=True,label=filename) )
            '''
            ###### CONNECT FFWD AND FBK POULATIONS TO OUT POPULATION ######
            # Populations of each joint are connected together according to connection
            # rules and network topology. There is no connections across joints.

            # Since in this simple version there is no feedback, we can also omit the "out" node

            '''
            self.ffwd_p[i].connect(self.out_p[i], rule='one_to_one', w= params['wgt_ffwd_out'], d=res)
            self.ffwd_p[i].connect(self.out_n[i], rule='one_to_one', w= params['wgt_ffwd_out'], d=res)
            self.ffwd_n[i].connect(self.out_p[i], rule='one_to_one', w=-params['wgt_ffwd_out'], d=res)
            self.ffwd_n[i].connect(self.out_n[i], rule='one_to_one', w=-params['wgt_ffwd_out'], d=res)
            
            self.fbk_p[i].connect(self.out_p[i], rule='one_to_one', w= params['wgt_fbk_out'], d=res)
            self.fbk_p[i].connect(self.out_n[i], rule='one_to_one', w= params['wgt_fbk_out'], d=res)
            self.fbk_n[i].connect(self.out_p[i], rule='one_to_one', w=-params['wgt_fbk_out'], d=res)
            self.fbk_n[i].connect(self.out_n[i], rule='one_to_one', w=-params['wgt_fbk_out'], d=res)
            '''
