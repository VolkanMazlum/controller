"""Cerebellum class"""

__authors__ = "Massimo Grillo"
__copyright__ = "Copyright 2021"
__credits__ = ["Massimo Grillo"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt
import nest
import trajectories as tj
from population_view import PopView
import mpi4py

from bsb.core import from_storage
#from bsb.output import HDF5Formatter
#from bsb.core import HDF5Formatter
#from bsb.config import JSONConfig
#from bsb.reporting import set_verbosity

class Cerebellum:

    #def __init__(self, filename_h5, filename_config, numNeurons, time_vect, traj_joint, plant, pathData="./data/", precise=False, **kwargs):
    def __init__(self, filename_h5, filename_config, multi = False, suffix = ''):

        # Reconfigure scaffold
        self.filename_h5 = filename_h5
        self.filename_config = filename_config
        self.suffix = suffix
        self.multi = multi
        
        '''
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            reconfigured_obj = JSONConfig(filename_config)
            HDF5Formatter.reconfigure(filename_h5, reconfigured_obj)
            reconfigured = True
            print(mpi4py.MPI.COMM_WORLD.rank)
        else:
            print(mpi4py.MPI.COMM_WORLD.rank)
            reconfigured = None
        print('mi blocco qui')
        # I can not use the barrier, otherwise the rank controlling music would stop the code here
        mpi4py.MPI.COMM_WORLD.bcast(reconfigured, root=0)
        print('ok')
        '''

        # Create scaffold_model from HDF5
        self.scaffold_model = from_storage(filename_h5)
        #set_verbosity(3)
        '''
        # Create adapter
        self.tuning_adapter = self.scaffold_model.create_adapter("tuning_weights") #tuning_weights
        if multi:
            print("Multi instance enabled")
            self.tuning_adapter.enable_multi(suffix)
        self.tuning_adapter.prepare()
        '''
        # Find ids for each cell type
        self.find_cells()

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
        print(self.scaffold_model.get_placement_sets())
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
