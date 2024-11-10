"""Cerebellum class"""

__authors__ = "Massimo Grillo"
__copyright__ = "Copyright 2021"
__credits__ = ["Massimo Grillo"]
__license__ = "GPL"
__version__ = "1.0.1"

import nest
import numpy as np
import matplotlib.pyplot as plt

import trajectories as tj
from population_view import PopView
import music
from mpi4py import MPI


#from bsb.core import from_storage
from bsb import options, ConfigurationError, from_storage, SimulationData
from bsb_nest import NestAdapter
from bsb_nest.adapter import NestResult
from settings import Simulation, Experiment
sim = Simulation()
exp = Experiment()
res = sim.resolution
pathData = exp.pathData + "nest/"

class Cerebellum:

    #def __init__(self, filename_h5, filename_config, numNeurons, time_vect, traj_joint, plant, pathData="./data/", precise=False, **kwargs):
    def __init__(self, filename_h5, filename_config, multi = False, suffix = ''):
        print("init cerebellum")
        # Reconfigure scaffold
        adapter = NestAdapter()
        print("adapter")
        options.verbosity = 4
        self.filename_h5 = filename_h5
        self.filename_config = filename_config
        self.suffix = suffix
        self.multi = multi
        self.forward_model = None
        
        world = MPI.COMM_WORLD
        if world.Get_rank() != 3:
            group = world.group.Excl([3])
            comm = world.Create_group(group)
            self.forward_model = from_storage("mouse_cereb_microzones_nest.hdf5", comm)
            self.inverse_model = from_storage("mouse_cereb_microzones_nest2.hdf5", comm)
        
        print("model created")
        
        #self.inverse_model = from_storage(filename_h5)
        
        simulation_name = "basal_activity"
        simulation_forw = self.forward_model.get_simulation(simulation_name)
        simulation_inv = self.inverse_model.get_simulation(simulation_name)
        adapter.simdata[simulation_forw] = SimulationData(simulation_forw, result=NestResult(simulation_forw))
        adapter.simdata[simulation_inv] = SimulationData(simulation_inv, result=NestResult(simulation_inv))
        # At some point in BSB this script the kernel is reset, so we need to load the module and set the parameters before the cerebellar nodes are created
        nest.Install("controller_module") 
        nest.SetKernelStatus({"resolution": res})
        nest.SetKernelStatus({"overwrite_files": True})
        nest.SetKernelStatus({"data_path": pathData})
        
        adapter.load_modules(simulation_forw)
        adapter.load_modules(simulation_inv)

        adapter.set_settings(simulation_forw)
        adapter.set_settings(simulation_inv)

        simulation_forw_duration = simulation_forw.duration 
        simulation_inv_duration = simulation_inv.duration 
        
        simulation_forw_resolution = simulation_forw.resolution
        simulation_inv_resolution = simulation_inv.resolution 
        print('duration, ', simulation_forw_duration)
        print('resolution, ', simulation_forw_resolution)
        adapter.create_neurons(simulation_forw)
        adapter.create_neurons(simulation_inv)
        print("created neurons")
        
        
        for neuron_model, gids in adapter.simdata[simulation_forw].populations.items():
            print('forward', neuron_model.name, gids)
        
        for neuron_model, gids in adapter.simdata[simulation_inv].populations.items():
            print('inverse', neuron_model.name, gids)
        
        if world.Get_rank() != 3:
            group = world.group.Excl([3])
            comm = world.Create_group(group)
            adapter.connect_neurons(simulation_forw)
            adapter.connect_neurons(simulation_inv)
        '''
        adapter.connect_neurons(simulation_forw)
        print("neurons connected")
        

        adapter.create_devices(simulation_forw)
        adapter.create_devices(simulation_inv)
        '''
        print("neurons connected")
        '''
        for device_model, device_ids in adapter.simdata[simulation_forw].devices.items():
            print(device_model.name, device_ids)
        '''
        ### FORWARD MODEL
        # Mossy fibers
        self.forw_Nest_Mf = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "mossy_fibers"][0]
        # Basket cells
        self.forw_N_BC = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "basket_cell"][0]
        # Stellate cells
        self.forw_N_SC = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "stellate_cell"][0]
        # IO
        self.forw_N_IO_plus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "io_plus"][0]
        self.forw_N_IO_minus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "io_minus"][0]
        # DCN
        self.forw_N_DCNp_plus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "dcn_p_plus"][0]
        self.forw_N_DCNp_minus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "dcn_p_minus"][0]
        self.forw_N_DCNi_plus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "dcn_i_plus"][0]
        self.forw_N_DCNi_minus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "dcn_i_minus"][0]
        # PC
        self.forw_N_PC = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "purkinje_cell"][0]
        self.forw_N_PC_minus = [gids for neuron_model, gids in adapter.simdata[simulation_forw].populations.items() if neuron_model.name == "purkinje_cell_minus"][0]
        '''
        ### INVERSE MODEL
        # Mossy fibers
        self.inv_Nest_Mf = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "mossy_fibers"][0]
        # Basket cells
        self.inv_N_BC = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "basket_cell"][0]
        # Stellate cells
        self.inv_N_SC = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "stellate_cell"][0]
        # IO
        self.inv_N_IO_plus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "io_plus"][0]
        self.inv_N_IO_minus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "io_minus"][0]
        # DCN
        self.inv_N_DCNp_plus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "dcn_p_plus"][0]
        self.inv_N_DCNp_minus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "dcn_p_minus"][0]
        self.inv_N_DCNi_plus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "dcn_i_plus"][0]
        self.inv_N_DCNi_minus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "dcn_i_minus"][0]
        # PC
        self.inv_N_PC = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "purkinje_cell"][0]
        self.inv_N_PC_minus = [gids for neuron_model, gids in adapter.simdata[simulation_inv].populations.items() if neuron_model.name == "purkinje_cell_minus"][0]
        # Find ids for each cell type
        #self.find_cells()
	'''
    def find_cells(self):
        '''
        if self.multi:
            suffix = self.suffix + '_'
        else:
            suffix = ''
        '''
        # Find bsb ids
        self.S_GR = self.scaffold_model.get_placement_set("granule_cell")
        
        self.S_Go = self.scaffold_model.get_placement_set("golgi_cell")
        #self.S_DCN = self.scaffold_model.get_placement_set("dcn_cell_glut_large")
        self.S_PC = self.scaffold_model.get_placement_set("purkinje_cell")
        self.S_BC = self.scaffold_model.get_placement_set("basket_cell")
        self.S_SC = self.scaffold_model.get_placement_set("stellate_cell")
        #self.S_DCN_GABA = self.scaffold_model.get_placement_set("dcn_cell_GABA")
        self.S_Mf = self.scaffold_model.get_placement_set("mossy_fibers")
        #print(self.scaffold_model.get_placement_sets())
        #self.S_IO = self.scaffold_model.get_placement_set("io_cell")
        '''
        # Subdivision into microzones
        uz_pos = self.scaffold_model.labels["microzone-positive"]
        uz_neg = self.scaffold_model.labels["microzone-negative"]
        S_IOp = np.intersect1d(self.S_IO, uz_pos)
        S_IOn = np.intersect1d(self.S_IO, uz_neg)
        S_DCNp = np.intersect1d(self.S_DCN, uz_pos)
        S_DCNn = np.intersect1d(self.S_DCN, uz_neg)
        S_PCp = np.intersect1d(self.S_PC, uz_pos)
        S_PCn = np.intersect1d(self.S_PC, uz_neg)
        S_BCp,S_BCn = self.subdivide_bc(S_PCn, S_PCp, S_IOn, S_IOp)
        S_SCp,S_SCn = self.subdivide_sc(S_PCn, S_PCp)

        # Transform into Nest ids
        self.Nest_Mf = self.tuning_adapter.get_nest_ids(self.S_Mf)
        self.io_neurons = self.tuning_adapter.get_nest_ids(self.S_IO)
        N_BCp = self.tuning_adapter.get_nest_ids(S_BCp)
        N_BCn = self.tuning_adapter.get_nest_ids(S_BCn)
        N_SCp = self.tuning_adapter.get_nest_ids(S_SCp)
        N_SCn = self.tuning_adapter.get_nest_ids(S_SCn)
        self.N_IOp = self.tuning_adapter.get_nest_ids(S_IOp)
        self.N_IOn = self.tuning_adapter.get_nest_ids(S_IOn)
        self.N_DCNp = self.tuning_adapter.get_nest_ids(S_DCNp)
        self.N_DCNn = self.tuning_adapter.get_nest_ids(S_DCNn)
        N_PCp = self.tuning_adapter.get_nest_ids(S_PCp)
        N_PCn = self.tuning_adapter.get_nest_ids(S_PCn)

        self.Nest_ids = {
            suffix + "dcn_cell_glut_large":{"positive":self.N_DCNp, "negative":self.N_DCNn},
            suffix + "purkinje_cell":{"positive":N_PCp, "negative":N_PCn},
            suffix + "basket_cell":{"positive":N_BCp, "negative":N_BCn},
            suffix + "stellate_cell":{"positive":N_SCp, "negative":N_SCn},
            suffix + "io_cell":{"positive":self.N_IOp, "negative":self.N_IOn},
        
        }
        '''

    def subdivide_bc(self, S_PCn, S_PCp, S_IOn, S_IOp):
        basket_to_pc = self.scaffold_model.get_connectivity_set("basket_to_purkinje")
        basket = np.unique(basket_to_pc.from_identifiers)
        basket_tot = basket_to_pc.from_identifiers
        pc_tot = basket_to_pc.to_identifiers
        S_BCp = []
        S_BCn = []
        N_pos = []
        N_neg = []
        for bc_id in basket:
            #simple_spikes = list(ts_pc[np.where(ts_pc < first_element)[0]])
            pc_ids = [j for i,j in enumerate(pc_tot) if basket_tot[i]==bc_id]
            count_pos = 0
            count_neg = 0
            for pc_id in pc_ids:
                if pc_id in S_PCp:
                    count_pos+=1
                elif pc_id in S_PCn:
                    count_neg+=1
                else:
                    print('strano')
            N_pos.append(count_pos/ len(pc_ids))
            N_neg.append(count_neg / len(pc_ids))
            if count_pos > count_neg:
                S_BCp.append(bc_id)
            else:
                S_BCn.append(bc_id)
        # Add also BCs not connected to PCs
        for bc in self.S_BC:
            if bc not in S_BCp and bc not in S_BCn:
                S_BCp.append(bc)
                S_BCn.append(bc)

        io_to_basket = self.scaffold_model.get_connectivity_set("io_to_basket")
        io = np.unique(io_to_basket.from_identifiers)
        io_tot = io_to_basket.from_identifiers
        basket = np.unique(io_to_basket.to_identifiers)
        basket_tot = io_to_basket.to_identifiers
        N_pos = []
        N_neg = []
        for bc_id in basket:
            #simple_spikes = list(ts_pc[np.where(ts_pc < first_element)[0]])
            io_ids = [j for i,j in enumerate(io_tot) if basket_tot[i]==bc_id]
            count_pos = 0
            count_neg = 0
            for io_id in io_ids:
                if io_id in S_IOp:
                    count_pos+=1
                elif io_id in S_IOn:
                    count_neg+=1
                else:
                    print('strano')
            N_pos.append(count_pos/ len(io_ids))
            N_neg.append(count_neg / len(io_ids))
        return S_BCp,S_BCn


    def subdivide_sc(self, S_PCn, S_PCp):
        stellate_to_pc = self.scaffold_model.get_connectivity_set("stellate_to_purkinje")
        stellate = np.unique(stellate_to_pc.from_identifiers)
        stellate_tot = stellate_to_pc.from_identifiers
        pc_tot = stellate_to_pc.to_identifiers
        S_SCp = []
        S_SCn = []
        for sc_id in stellate:
            pc_ids = [j for i,j in enumerate(pc_tot) if stellate_tot[i]==sc_id]
            count_pos = 0
            count_neg = 0
            for pc_id in pc_ids:
                if pc_id in S_PCp:
                    count_pos+=1
                elif pc_id in S_PCn:
                    count_neg+=1
                else:
                    print('strano')
            if count_pos > count_neg:
                S_SCp.append(sc_id)
            else:
                S_SCn.append(sc_id)
        # Add also SCs not connected to PCs
        for sc in self.S_SC:
            if sc not in S_SCp and sc not in S_SCn:
                S_SCp.append(sc)
                S_SCn.append(sc)
        return S_SCp,S_SCn
