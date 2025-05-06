"""Cerebellum class"""

__authors__ = "Massimo Grillo"
__copyright__ = "Copyright 2021"
__credits__ = ["Massimo Grillo"]
__license__ = "GPL"
__version__ = "1.0.1"

from pathlib import Path

import matplotlib.pyplot as plt
import music
import nest
import numpy as np

# from bsb.core import from_storage
from bsb import (
    ConfigurationError,
    Scaffold,
    SimulationData,
    config,
    from_storage,
    get_simulation_adapter,
    options,
)
from bsb_nest import NestAdapter
from bsb_nest.adapter import NestResult
from mpi4py import MPI
from settings import Experiment, Simulation

import trajectories as tj
from population_view import PopView

# maybe these can be moved in a paths object together with other folders!
SIMULATION_NAME_IN_YAML = "basal_activity"
PATH_HDF5 = Path("/sim/controller/cerebellum/mouse_cereb_microzones_complete.hdf5")
PATH_YAML_FORWARD = Path("/sim/controller/conf/forward.yaml")
PATH_YAML_INVERSE = Path("/sim/controller/conf/inverse.yaml")


class Cerebellum:

    def __init__(
        self,
        comm: MPI.Comm,
        filename_h5,
        filename_config,
    ):
        print("init cerebellum")
        options.verbosity = 4
        print(f"increased bsb verbosity to {options.verbosity}")
        self.filename_h5 = filename_h5
        self.filename_config = filename_config
        self.forward_model = None

        adapter: NestAdapter = get_simulation_adapter("nest", comm)
        # hdf5 uses relative paths from itself to find functions, so if we move it it won't work anymore

        self.forward_model = from_storage(str(PATH_HDF5), comm)
        conf_forward = config.parse_configuration_file(str(PATH_YAML_FORWARD))
        self.forward_model.simulations[SIMULATION_NAME_IN_YAML] = (
            conf_forward.simulations[SIMULATION_NAME_IN_YAML]
        )
        print("loaded forward model and its configuration")

        self.inverse_model = from_storage(str(PATH_HDF5), comm)
        conf_inverse = config.parse_configuration_file(str(PATH_YAML_INVERSE))
        self.inverse_model.simulations[SIMULATION_NAME_IN_YAML] = (
            conf_inverse.simulations[SIMULATION_NAME_IN_YAML]
        )
        print("loaded inverse model and its configuration")

        simulation_forw = self.forward_model.get_simulation(SIMULATION_NAME_IN_YAML)
        simulation_inv = self.inverse_model.get_simulation(SIMULATION_NAME_IN_YAML)

        adapter.simdata[simulation_forw] = SimulationData(
            simulation_forw, result=NestResult(simulation_forw)
        )
        adapter.simdata[simulation_inv] = SimulationData(
            simulation_inv, result=NestResult(simulation_inv)
        )

        adapter.load_modules(simulation_forw)
        adapter.load_modules(simulation_inv)

        adapter.set_settings(simulation_forw)
        adapter.set_settings(simulation_inv)

        print(f"duration: FWD:{simulation_forw.duration}; INV{simulation_inv.duration}")
        print(
            f"resolution: FWD:{simulation_forw.resolution}; INV{simulation_inv.resolution}"
        )

        adapter.create_neurons(simulation_forw)
        adapter.create_neurons(simulation_inv)
        print("created cerebellum neurons")

        adapter.connect_neurons(simulation_forw)
        adapter.connect_neurons(simulation_inv)
        print("connected cerebellum neurons")

        adapter.create_devices(simulation_forw)
        adapter.create_devices(simulation_inv)
        print("created cerebellum devices")

        ### FORWARD MODEL
        # Mossy fibers
        self.forw_Nest_Mf = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "mossy_fibers"
        )

        # Glomerulus
        self.forw_N_Glom = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "glomerulus"
        )
        # Granule cells
        self.forw_N_GrC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "granule_cell"
        )
        # Golgi cells
        self.forw_N_GoC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "golgi_cell"
        )
        # Basket cells
        self.forw_N_BC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "basket_cell"
        )
        # Stellate cells
        self.forw_N_SC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "stellate_cell"
        )
        # IO
        self.forw_N_IO_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "io_plus"
        )
        self.forw_N_IO_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "io_minus"
        )
        # DCN
        self.forw_N_DCNp_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_p_plus"
        )
        self.forw_N_DCNp_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_p_minus"
        )
        forw_N_DCNi_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_i_plus"
        )
        forw_N_DCNi_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_i_minus"
        )
        # PC
        self.forw_N_PC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "purkinje_cell_plus"
        )
        self.forw_N_PC_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "purkinje_cell_minus"
        )

        ### INVERSE MODEL
        # Mossy fibers
        self.inv_Nest_Mf = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "mossy_fibers"
        )
        # Glomerulus
        self.inv_N_Glom = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "glomerulus"
        )
        # Granule cells
        self.inv_N_GrC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "granule_cell"
        )
        # Golgi cells
        self.inv_N_GoC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "golgi_cell"
        )
        # Basket cells
        self.inv_N_BC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "basket_cell"
        )
        # Stellate cells
        self.inv_N_SC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "stellate_cell"
        )
        # IO
        self.inv_N_IO_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "io_plus"
        )
        self.inv_N_IO_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "io_minus"
        )
        # DCN
        self.inv_N_DCNp_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_p_plus"
        )
        self.inv_N_DCNp_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_p_minus"
        )
        inv_N_DCNi_plus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_i_plus"
        )
        inv_N_DCNi_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_i_minus"
        )
        # PC
        self.inv_N_PC = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "purkinje_cell_plus"
        )
        self.inv_N_PC_minus = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "purkinje_cell_minus"
        )
