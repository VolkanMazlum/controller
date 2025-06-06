import os
from pathlib import Path

import numpy as np
import structlog

# from bsb.core import from_storage
from bsb import SimulationData, config, from_storage, get_simulation_adapter, options
from bsb_nest import NestAdapter
from bsb_nest.adapter import NestResult
from mpi4py import MPI

from complete_control.config.bsb_models import BSBConfigPaths

from .CerebellumPopulations import CerebellumPopulations
from .population_view import PopView

SIMULATION_NAME_IN_YAML = "basal_activity"


class Cerebellum:
    def __init__(
        self,
        comm: MPI.Comm,
        paths: BSBConfigPaths,
        total_time_vect: np.ndarray,
        label_prefix: str,
    ):
        self.log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)
        options.verbosity = (
            3  # TODO how to we handle this verbosity? keep 0 for now but...
        )
        self.total_time_vect = total_time_vect
        self.label_prefix = label_prefix
        self.populations = CerebellumPopulations()
        self.forward_model = None

        self.log.warning(
            f"Loading Cerebellum with comm {comm} rank {comm.Get_rank()} and size {comm.Get_size()}")
        adapter: NestAdapter = get_simulation_adapter("nest", comm)
        # hdf5 uses relative paths from itself to find functions, so if we move it it won't work anymore

        self.forward_model = from_storage(str(paths.cerebellum_hdf5), comm)
        conf_forward = config.parse_configuration_file(str(paths.forward_yaml))
        self.forward_model.simulations[SIMULATION_NAME_IN_YAML] = (
            conf_forward.simulations[SIMULATION_NAME_IN_YAML]
        )
        self.log.debug("loaded forward model and its configuration")

        self.inverse_model = from_storage(str(paths.cerebellum_hdf5), comm)
        conf_inverse = config.parse_configuration_file(str(paths.inverse_yaml))
        self.inverse_model.simulations[SIMULATION_NAME_IN_YAML] = (
            conf_inverse.simulations[SIMULATION_NAME_IN_YAML]
        )
        self.log.debug("loaded inverse model and its configuration")

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

        # let's just keep the settings we set for the rest of the simulation. specifically, the seed
        # adapter.set_settings(simulation_forw)
        # adapter.set_settings(simulation_inv)

        self.log.debug(
            f"duration: FWD:{simulation_forw.duration}; INV{simulation_inv.duration}"
        )
        self.log.debug(
            f"resolution: FWD:{simulation_forw.resolution}; INV{simulation_inv.resolution}"
        )

        adapter.create_neurons(simulation_forw)
        adapter.create_neurons(simulation_inv)
        self.log.debug("created cerebellum neurons")

        adapter.connect_neurons(simulation_forw)
        adapter.connect_neurons(simulation_inv)
        self.log.debug("connected cerebellum neurons")

        adapter.create_devices(simulation_forw)
        adapter.create_devices(simulation_inv)
        self.log.debug("created cerebellum devices")

        ### FORWARD MODEL
        # Mossy fibers
        self.forw_Nest_Mf = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "mossy_fibers"
        )
        self.N_mossy_forw = int(len(self.forw_Nest_Mf) / 2)

        # Glomerulus
        _forw_N_Glom_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "glomerulus"
        )
        # Granule cells
        _forw_N_GrC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "granule_cell"
        )
        # Golgi cells
        _forw_N_GoC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "golgi_cell"
        )
        # Basket cells
        _forw_N_BC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "basket_cell"
        )
        # Stellate cells
        _forw_N_SC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "stellate_cell"
        )
        # IO
        _forw_N_IO_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "io_plus"
        )
        _forw_N_IO_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "io_minus"
        )
        # DCN
        _forw_N_DCNp_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_p_plus"
        )
        _forw_N_DCNp_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_p_minus"
        )
        _forw_N_DCNi_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_i_plus"
        )
        _forw_N_DCNi_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "dcn_i_minus"
        )
        # PC
        _forw_N_PC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_forw
            ].populations.items()
            if neuron_model.name == "purkinje_cell_plus"
        )
        _forw_N_PC_minus_gids = next(
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
        self.N_mossy_inv = int(len(self.inv_Nest_Mf) / 2)

        # Glomerulus
        _inv_N_Glom_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "glomerulus"
        )
        # Granule cells
        _inv_N_GrC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "granule_cell"
        )
        # Golgi cells
        _inv_N_GoC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "golgi_cell"
        )
        # Basket cells
        _inv_N_BC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "basket_cell"
        )
        # Stellate cells
        _inv_N_SC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "stellate_cell"
        )
        # IO
        _inv_N_IO_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "io_plus"
        )
        _inv_N_IO_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "io_minus"
        )
        # DCN
        _inv_N_DCNp_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_p_plus"
        )
        _inv_N_DCNp_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_p_minus"
        )
        _inv_N_DCNi_plus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_i_plus"
        )
        _inv_N_DCNi_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "dcn_i_minus"
        )
        # PC
        _inv_N_PC_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "purkinje_cell_plus"
        )
        _inv_N_PC_minus_gids = next(
            gids
            for neuron_model, gids in adapter.simdata[
                simulation_inv
            ].populations.items()
            if neuron_model.name == "purkinje_cell_minus"
        )

        self._create_core_pop_views(
            _forw_N_Glom_gids,
            _forw_N_GrC_gids,
            _forw_N_GoC_gids,
            _forw_N_BC_gids,
            _forw_N_SC_gids,
            _forw_N_IO_plus_gids,
            _forw_N_IO_minus_gids,
            _forw_N_DCNp_plus_gids,
            _forw_N_DCNp_minus_gids,
            _forw_N_PC_gids,
            _forw_N_PC_minus_gids,
            _inv_N_Glom_gids,
            _inv_N_GrC_gids,
            _inv_N_GoC_gids,
            _inv_N_BC_gids,
            _inv_N_SC_gids,
            _inv_N_IO_plus_gids,
            _inv_N_IO_minus_gids,
            _inv_N_DCNp_plus_gids,
            _inv_N_DCNp_minus_gids,
            _inv_N_PC_gids,
            _inv_N_PC_minus_gids,
        )

    def _create_core_pop_views(
        self,
        _forw_N_Glom_gids,
        _forw_N_GrC_gids,
        _forw_N_GoC_gids,
        _forw_N_BC_gids,
        _forw_N_SC_gids,
        _forw_N_IO_plus_gids,
        _forw_N_IO_minus_gids,
        _forw_N_DCNp_plus_gids,
        _forw_N_DCNp_minus_gids,
        _forw_N_PC_gids,
        _forw_N_PC_minus_gids,
        _inv_N_Glom_gids,
        _inv_N_GrC_gids,
        _inv_N_GoC_gids,
        _inv_N_BC_gids,
        _inv_N_SC_gids,
        _inv_N_IO_plus_gids,
        _inv_N_IO_minus_gids,
        _inv_N_DCNp_plus_gids,
        _inv_N_DCNp_minus_gids,
        _inv_N_PC_gids,
        _inv_N_PC_minus_gids,
    ):
        """Creates PopView instances for all core NEST populations."""
        # Forward Model PopViews
        self.populations.forw_mf_p_view = PopView(
            self.forw_Nest_Mf[-self.N_mossy_forw :],
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_mf_p",
        )
        self.populations.forw_mf_n_view = PopView(
            self.forw_Nest_Mf[0 : self.N_mossy_forw],
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_mf_n",
        )
        self.populations.forw_glom_view = PopView(
            _forw_N_Glom_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_glom",
        )
        self.populations.forw_grc_view = PopView(
            _forw_N_GrC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_grc",
        )
        self.populations.forw_goc_view = PopView(
            _forw_N_GoC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_goc",
        )
        self.populations.forw_bc_view = PopView(
            _forw_N_BC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_bc",
        )
        self.populations.forw_sc_view = PopView(
            _forw_N_SC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_sc",
        )
        self.populations.forw_io_p_view = PopView(
            _forw_N_IO_plus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_io_p",
        )
        self.populations.forw_io_n_view = PopView(
            _forw_N_IO_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_io_n",
        )
        self.populations.forw_dcnp_p_view = PopView(
            _forw_N_DCNp_plus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_dcnp_p",
        )
        self.populations.forw_dcnp_n_view = PopView(
            _forw_N_DCNp_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_dcnp_n",
        )
        self.populations.forw_pc_p_view = PopView(
            _forw_N_PC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_pc_p",
        )
        self.populations.forw_pc_n_view = PopView(
            _forw_N_PC_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}forw_pc_n",
        )

        # Inverse Model PopViews
        self.populations.inv_mf_p_view = PopView(
            self.inv_Nest_Mf[-self.N_mossy_inv :],
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_mf_p",
        )
        self.populations.inv_mf_n_view = PopView(
            self.inv_Nest_Mf[0 : self.N_mossy_inv],
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_mf_n",
        )
        self.populations.inv_glom_view = PopView(
            _inv_N_Glom_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_glom",
        )
        self.populations.inv_grc_view = PopView(
            _inv_N_GrC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_grc",
        )
        self.populations.inv_goc_view = PopView(
            _inv_N_GoC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_goc",
        )
        self.populations.inv_bc_view = PopView(
            _inv_N_BC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_bc",
        )
        self.populations.inv_sc_view = PopView(
            _inv_N_SC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_sc",
        )
        self.populations.inv_io_p_view = PopView(
            _inv_N_IO_plus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_io_p",
        )
        self.populations.inv_io_n_view = PopView(
            _inv_N_IO_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_io_n",
        )
        self.populations.inv_dcnp_p_view = PopView(
            _inv_N_DCNp_plus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_dcnp_p",
        )
        self.populations.inv_dcnp_n_view = PopView(
            _inv_N_DCNp_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_dcnp_n",
        )
        self.populations.inv_pc_p_view = PopView(
            _inv_N_PC_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_pc_p",
        )
        self.populations.inv_pc_n_view = PopView(
            _inv_N_PC_minus_gids,
            self.total_time_vect,
            to_file=True,
            label=f"{self.label_prefix}inv_pc_n",
        )
