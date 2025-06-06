from typing import Any, Dict, Optional

import nest
import numpy as np
import structlog
from config.bsb_models import BSBConfigPaths
from config.connection_params import ConnectionsParams
from config.core_models import SimulationParams
from config.population_params import PopulationsParams
from mpi4py.MPI import Comm

from .Cerebellum import Cerebellum
from .CerebellumHandlerPopulations import CerebellumHandlerPopulations  # Added import
from .ControllerPopulations import ControllerPopulations
from .population_view import PopView

#                                    ┌───┐
#                              ┌────▶│Mf └──────────────┐
#                              │     └───┐              └────┐    ┌───────────┐
#                              │         │Inverse model | DCN├───►│ motor     │
#                              │     ┌───┘              ┌────┘    │ prediction│
#                              │  ┌─▶│IO ┌──────────────┘         └────┬──────┘
#                              │  │  └───┘                             │
# t      ┌─────────┐           │  │                                    │
# r ────▶│ Planner │───┬───────┘  │          Motor Cortex              │
# a      └─────────┘   │          │ ┌───────────────────────────┐      ▼
# j.txt                │          │ │  ┌──────────┐  ┌─────────┐│   ┌─────────┐
#                      │+         │ │  │   Ffwd   ├─►│   Out   ├┼─┬►│Smoothing│
#                      └──►█──────┤ │  └──────────┘  └─────────┘│ │ └─────────┘         __
#                          ▲      │ │                     ▲     │ │       |     _(\    |@@|
#                        - │      │ │                     │     │ │       ▼    (__/\__ \--/ __
#                          │      │ │  ┌──────────┐       │     │ │    robotic    \___|----|  |   __
#                          │      └─┼─►│   Fbk    ├───────┘     │ │     plant         \ }{ /\ )_ / _\
#                      ┌───┘        │  └──────────┘             │ │       │           /\__/\ \__O (__
#                      │            └───────────────────────────┘ │       ▼          (--/\--)    \__/
#           ┌──────────┴──┐     ⚖️                                │ ┌───────────┐    _)(  )(_
#           │  State      │◄────█◄────────────────────────────────)┬┤  Sensory  │     --''---`
#           │  estimator  │     ▲                      ┌───┐      │││   system  │
#           └─────────────┘     │        ┌─────────────┘ Mf│◄─────┘│└───────────┘
#                               │    ┌───┘              ┌──┘       │
#                               ├────┤DCN| Forward model│          │
#                               │    └───┐              └──┐       │
#                               │        └──────────────┐IO│◄────█◄┘+
#                               │                       └──┘     ▲
#                               └────────────────────────────────┘-
#


class CerebellumHandler:
    """
    Encapsulates the NEST network components and connections for the cerebellum model
    and its interface populations, designed to be instantiated within Controller.
    """

    def __init__(
        self,
        N: int,
        total_time_vect: np.ndarray,
        sim_params: SimulationParams,
        pops_params: PopulationsParams,  # Parameters for interface populations
        conn_params: ConnectionsParams,
        cerebellum_paths: BSBConfigPaths,  # Params for Cerebellum object (paths etc)
        path_data: str,
        comm: Comm,
        controller_pops: Optional[ControllerPopulations],
        label_prefix: str = "cereb_",
        dof_id: int = 0,
    ):
        """
        Initializes the Cerebellum controller module.

        Args:
            N (int): Number of neurons per sub-population in interface layers.
            total_time_vect (np.ndarray): Simulation time vector.
            sim_params (Dict[str, Any]): General simulation parameters (e.g., res).
            pops_params (Dict[str, Any]): Parameters for interface populations.
            conn_params (Dict[str, Any]): Parameters for connections.
            cerebellum_config (Dict[str, Any]): Configuration for the Cerebellum build.
            path_data (str): Path for NEST data output.
            label_prefix (str): Prefix for PopView labels.
            dof_id (int): Degree of freedom identifier (currently unused internally).
            controller_pops (Optional[ControllerPopulations]): Populations from the main controller.
        """
        self.log = structlog.get_logger(f"cerebellum_controller.dof_{dof_id}")
        self.log.info("Initializing CerebellumHandler")

        self.N = N
        self.total_time_vect = total_time_vect
        self.sim_params = sim_params
        self.pops_params = pops_params
        self.conn_params = conn_params
        self.cerebellum_config = cerebellum_paths
        self.path_data = path_data
        self.comm = comm
        self.label_prefix = label_prefix
        self.res = sim_params.resolution
        self.controller_pops = controller_pops

        # --- Initialize Interface Populations Dataclass ---
        self.interface_pops = CerebellumHandlerPopulations()

        # --- Instantiate the Core Cerebellum Model ---
        self.log.info("Instantiating core Cerebellum object", config=cerebellum_paths)
        self.cerebellum = Cerebellum(
            comm=comm,
            paths=cerebellum_paths,
            total_time_vect=self.total_time_vect,
            label_prefix=f"{self.label_prefix}core_",
        )
        self.log.info("Core Cerebellum object instantiated.")
        # Get N_mossy counts from the Cerebellum object
        self.N_mossy_forw = self.cerebellum.N_mossy_forw
        self.N_mossy_inv = self.cerebellum.N_mossy_inv
        self.log.info(
            "Mossy fiber counts", N_forw=self.N_mossy_forw, N_inv=self.N_mossy_inv
        )

        # --- Create Interface Populations ---
        self.log.info("Creating interface populations")
        self._create_interface_populations()

        # --- Connect Interface Populations to Core Cerebellum ---
        self.log.info("Connecting interface populations to core cerebellum")
        self._connect_interfaces_to_core()

        # --- Connect Populations for Error Calculation ---
        self.log.info("Connecting populations for error calculation")
        self._connect_error_calculation()

        self.log.info("CerebellumHandler initialization complete.")

    def _create_pop_view(
        self, nest_pop: nest.NodeCollection, base_label: str
    ) -> PopView:
        """Helper to create PopView instance."""
        full_label = f"{self.label_prefix}{base_label}"
        # Use path_data implicitly if to_file=True
        return PopView(nest_pop, self.total_time_vect, to_file=True, label=full_label)

    def _create_interface_populations(self):
        """Creates the intermediate populations connecting to the cerebellum."""
        # --- Populations based on brain.py logic ---

        # Feedback Scaling (Input to Fwd Error Calc)
        # prediction_p and prediction_n are now created in Controller
        params = self.pops_params.feedback
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        feedback_p = nest.Create(
            "basic_neuron_nestml", self.N
        )  # Using basic_neuron like brain.py
        nest.SetStatus(feedback_p, {**pop_params, "pos": True})
        self.interface_pops.feedback_p = self._create_pop_view(feedback_p, "feedback_p")
        feedback_n = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(feedback_n, {**pop_params, "pos": False})
        self.interface_pops.feedback_n = self._create_pop_view(feedback_n, "feedback_n")

        # Motor Commands Relay (Input to Fwd MFs) - Size N_mossy_forw
        params = self.pops_params.motor_commands
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        motor_commands_p = nest.Create("basic_neuron_nestml", self.N_mossy_forw)
        nest.SetStatus(motor_commands_p, {**pop_params, "pos": True})
        self.interface_pops.motor_commands_p = self._create_pop_view(
            motor_commands_p, "motor_commands_p"
        )
        motor_commands_n = nest.Create("basic_neuron_nestml", self.N_mossy_forw)
        nest.SetStatus(motor_commands_n, {**pop_params, "pos": False})
        self.interface_pops.motor_commands_n = self._create_pop_view(
            motor_commands_n, "motor_commands_n"
        )

        # Forward Error Calculation (Input to Fwd IO)
        params = self.pops_params.error
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        error_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(error_p, {**pop_params, "pos": True})
        self.interface_pops.error_p = self._create_pop_view(error_p, "error_p")
        error_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(error_n, {**pop_params, "pos": False})
        self.interface_pops.error_n = self._create_pop_view(error_n, "error_n")

        # Planner Relay (Input to Inv MFs) - Size N_mossy_inv
        params = self.pops_params.plan_to_inv
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        plan_to_inv_p = nest.Create("basic_neuron_nestml", self.N_mossy_inv)
        nest.SetStatus(plan_to_inv_p, {**pop_params, "pos": True})
        self.interface_pops.plan_to_inv_p = self._create_pop_view(
            plan_to_inv_p, "plan_to_inv_p"
        )
        plan_to_inv_n = nest.Create("basic_neuron_nestml", self.N_mossy_inv)
        nest.SetStatus(plan_to_inv_n, {**pop_params, "pos": False})
        self.interface_pops.plan_to_inv_n = self._create_pop_view(
            plan_to_inv_n, "plan_to_inv_n"
        )

        # State Estimator Relay (Input to Inv Error Calc) - Size N_mossy_inv? Check brain.py usage
        # Assuming size N_mossy_inv based on plan_to_inv, adjust if needed
        # TODO why is this plan instead of state?
        params = self.pops_params.plan_to_inv
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        state_to_inv_p = nest.Create("basic_neuron_nestml", self.N_mossy_inv)
        nest.SetStatus(state_to_inv_p, {**pop_params, "pos": True})
        self.interface_pops.state_to_inv_p = self._create_pop_view(
            state_to_inv_p, "state_to_inv_p"
        )
        state_to_inv_n = nest.Create("basic_neuron_nestml", self.N_mossy_inv)
        nest.SetStatus(state_to_inv_n, {**pop_params, "pos": False})
        self.interface_pops.state_to_inv_n = self._create_pop_view(
            state_to_inv_n, "state_to_inv_n"
        )

        # Inverse Error Calculation (Input to Inv IO)
        params = self.pops_params.error_i
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        error_inv_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(error_inv_p, {**pop_params, "pos": True})
        self.interface_pops.error_inv_p = self._create_pop_view(
            error_inv_p, "error_inv_p"
        )
        error_inv_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(error_inv_n, {**pop_params, "pos": False})
        self.interface_pops.error_inv_n = self._create_pop_view(
            error_inv_n, "error_inv_n"
        )

        # Motor Prediction Scaling (Output from Inv DCN)
        params = self.pops_params.motor_pred
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        motor_prediction_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(motor_prediction_p, {**pop_params, "pos": True})
        self.interface_pops.motor_prediction_p = self._create_pop_view(
            motor_prediction_p, "motor_prediction_p"
        )
        motor_prediction_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(motor_prediction_n, {**pop_params, "pos": False})
        self.interface_pops.motor_prediction_n = self._create_pop_view(
            motor_prediction_n, "motor_prediction_n"
        )

        # Feedback Inverse Scaling (Input to Inv Error Calc?) - Check necessity
        params = self.pops_params.feedback_inv
        pop_params = {
            "kp": params.kp,
            "buffer_size": params.buffer_size,
            "base_rate": params.base_rate,
            "simulation_steps": len(self.total_time_vect),
        }
        feedback_inv_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(feedback_inv_p, {**pop_params, "pos": True})
        self.interface_pops.feedback_inv_p = self._create_pop_view(
            feedback_inv_p, "feedback_inv_p"
        )
        feedback_inv_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(feedback_inv_n, {**pop_params, "pos": False})
        self.interface_pops.feedback_inv_n = self._create_pop_view(
            feedback_inv_n, "feedback_inv_n"
        )

    def _connect_interfaces_to_core(self):
        """Connects interface populations to the core cerebellum model."""
        self.log.debug("Connecting interfaces to core cerebellum")

        # --- Forward Model Connections ---
        # Motor Commands -> Fwd Mossy Fibers
        self.log.debug("Connecting motor_commands -> fwd_mf")
        nest.Connect(
            self.interface_pops.motor_commands_p.pop,
            self.cerebellum.populations.forw_mf_p_view.pop,
            "one_to_one",
            # TODO no weight given
            # syn_spec={"weight": 1.0},
        )
        nest.Connect(
            self.interface_pops.motor_commands_n.pop,
            self.cerebellum.populations.forw_mf_n_view.pop,
            "one_to_one",
            # TODO no weight given
            # syn_spec={"weight": 1.0},
        )

        # Fwd Error -> Fwd Inferior Olive
        conn_spec_error_io_f = self.conn_params.error_io_f
        self.log.debug("Connecting error -> fwd_io", conn_spec=conn_spec_error_io_f)
        nest.Connect(
            self.interface_pops.error_p.pop,
            self.cerebellum.populations.forw_io_p_view.pop,
            "all_to_all",
            syn_spec=conn_spec_error_io_f.model_dump(exclude_none=True),
        )
        # Check sign for negative connection
        conn_spec_error_io_f_neg = conn_spec_error_io_f.model_copy(
            update={"weight": -conn_spec_error_io_f.weight}
        )
        nest.Connect(
            self.interface_pops.error_n.pop,
            self.cerebellum.populations.forw_io_n_view.pop,
            "all_to_all",
            syn_spec=conn_spec_error_io_f_neg.model_dump(exclude_none=True),
        )

        # --- Inverse Model Connections ---
        # Planner -> Inv Mossy Fibers
        self.log.debug("Connecting plan_to_inv -> inv_mf")
        nest.Connect(
            self.interface_pops.plan_to_inv_p.pop,
            self.cerebellum.populations.inv_mf_p_view.pop,
            "one_to_one",
            # TODO hello? what is this weight?
            # syn_spec={"weight": 1.0},
        )
        nest.Connect(
            self.interface_pops.plan_to_inv_n.pop,
            self.cerebellum.populations.inv_mf_n_view.pop,
            "one_to_one",
            # TODO hello? what is this weight? Check weight sign
            # syn_spec={"weight": 1.0},
        )

        # Inv Error -> Inv Inferior Olive
        conn_spec_error_inv_io_i = self.conn_params.error_inv_io_i
        self.log.debug(
            "Connecting error_inv -> inv_io", conn_spec=conn_spec_error_inv_io_i
        )
        nest.Connect(
            self.interface_pops.error_inv_p.pop,
            self.cerebellum.populations.inv_io_p_view.pop,
            "all_to_all",
            syn_spec=conn_spec_error_inv_io_i.model_dump(exclude_none=True),
        )
        # Assuming same sign convention for negative side
        # conn_spec_error_inv_io_i_neg = conn_spec_error_inv_io_i.model_copy(update={"weight": -conn_spec_error_inv_io_i.weight}) # TODO: Confirm if negative weight is needed
        nest.Connect(
            self.interface_pops.error_inv_n.pop,
            self.cerebellum.populations.inv_io_n_view.pop,
            "all_to_all",
            syn_spec=conn_spec_error_inv_io_i.model_dump(
                exclude_none=True
            ),  # TODO: Confirm weight sign logic, using original for now
        )

        # Inv DCN -> Motor Prediction Scaling Population
        conn_spec_dcn_i_mp = self.conn_params.dcn_i_motor_pred
        w = conn_spec_dcn_i_mp.weight
        d = conn_spec_dcn_i_mp.delay
        self.log.debug("Connecting inv_dcn -> motor_prediction", weight=w, delay=d)
        nest.Connect(
            self.cerebellum.populations.inv_dcnp_p_view.pop,
            self.interface_pops.motor_prediction_p.pop,
            "all_to_all",
            syn_spec={"weight": w, "delay": d},
        )
        nest.Connect(
            self.cerebellum.populations.inv_dcnp_p_view.pop,
            self.interface_pops.motor_prediction_n.pop,
            "all_to_all",
            syn_spec={"weight": w, "delay": d},
        )
        nest.Connect(
            self.cerebellum.populations.inv_dcnp_n_view.pop,
            self.interface_pops.motor_prediction_p.pop,
            "all_to_all",
            syn_spec={"weight": -w, "delay": d},
        )
        nest.Connect(
            self.cerebellum.populations.inv_dcnp_n_view.pop,
            self.interface_pops.motor_prediction_n.pop,
            "all_to_all",
            syn_spec={"weight": -w, "delay": d},
        )

    def _connect_error_calculation(self):
        """Connects populations involved in calculating error signals for IO."""
        self.log.debug("Connecting populations for error calculation")

        # --- Forward Error Calculation (Error = Feedback - Fwd_DCN_Prediction) ---
        # Connect Feedback -> Error
        conn_spec_fb_err = self.conn_params.feedback_error
        w_fb = conn_spec_fb_err.weight
        self.log.debug("Connecting feedback -> error", weight=w_fb)
        nest.Connect(
            self.interface_pops.feedback_p.pop,
            self.interface_pops.error_p.pop,
            "all_to_all",
            syn_spec={"weight": w_fb},
        )
        nest.Connect(
            self.interface_pops.feedback_p.pop,
            self.interface_pops.error_n.pop,
            "all_to_all",
            syn_spec={"weight": w_fb},
        )
        nest.Connect(
            self.interface_pops.feedback_n.pop,
            self.interface_pops.error_p.pop,
            "all_to_all",
            syn_spec={"weight": -w_fb},
        )
        nest.Connect(
            self.interface_pops.feedback_n.pop,
            self.interface_pops.error_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_fb},
        )

        # Connect Fwd DCN -> Error (Inhibitory)
        conn_spec_dcn_f_err = self.conn_params.dcn_f_error
        w_dcn = conn_spec_dcn_f_err.weight
        self.log.debug("Connecting fwd_dcn -> error (inhibitory)", weight=w_dcn)
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_p_view.pop,
            self.interface_pops.error_p.pop,
            "all_to_all",
            syn_spec={"weight": -w_dcn},
        )
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_p_view.pop,
            self.interface_pops.error_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_dcn},
        )
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_n_view.pop,
            self.interface_pops.error_p.pop,
            "all_to_all",
            syn_spec={"weight": w_dcn},
        )
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_n_view.pop,
            self.interface_pops.error_n.pop,
            "all_to_all",
            syn_spec={"weight": w_dcn},
        )

        # --- Inverse Error Calculation (Error = Plan - StateEst?) ---
        # Connect Plan -> Inv Error
        conn_spec_plan_err_inv = self.conn_params.plan_to_inv_error_inv
        w_plan = conn_spec_plan_err_inv.weight
        d_plan = conn_spec_plan_err_inv.delay
        self.log.debug(
            "Connecting plan_to_inv -> error_inv", weight=w_plan, delay=d_plan
        )
        nest.Connect(
            self.interface_pops.plan_to_inv_p.pop,
            self.interface_pops.error_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": w_plan, "delay": d_plan},
        )
        nest.Connect(
            self.interface_pops.plan_to_inv_p.pop,
            self.interface_pops.error_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": w_plan, "delay": d_plan},
        )
        nest.Connect(
            self.interface_pops.plan_to_inv_n.pop,
            self.interface_pops.error_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": -w_plan, "delay": d_plan},
        )
        nest.Connect(
            self.interface_pops.plan_to_inv_n.pop,
            self.interface_pops.error_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_plan, "delay": d_plan},
        )

        # Connect StateEst -> Inv Error (Inhibitory?)
        # TODO why is this called "plan" when it is the state? Using same spec for now.
        conn_spec_state_err_inv = self.conn_params.plan_to_inv_error_inv
        w_state = conn_spec_state_err_inv.weight
        d_state = conn_spec_state_err_inv.delay
        self.log.debug(
            "Connecting state_to_inv -> error_inv (inhibitory?)",
            weight=w_state,
            delay=d_state,
        )
        nest.Connect(
            self.interface_pops.state_to_inv_p.pop,
            self.interface_pops.error_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": w_state, "delay": d_state},
        )
        nest.Connect(
            self.interface_pops.state_to_inv_p.pop,
            self.interface_pops.error_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": w_state, "delay": d_state},
        )
        nest.Connect(
            self.interface_pops.state_to_inv_n.pop,
            self.interface_pops.error_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": -w_state, "delay": d_state},
        )
        nest.Connect(
            self.interface_pops.state_to_inv_n.pop,
            self.interface_pops.error_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_state, "delay": d_state},
        )

    def connect_to_main_controller_populations(self):
        if not self.controller_pops:
            self.log.error(
                "ControllerPopulations not provided, cannot connect to main controller."
            )
            raise ValueError(
                "ControllerPopulations not provided, cannot connect to main controller."
            )

        self.log.info("Connecting CerebellumHandler to main controller populations")

        # --- Connections FROM Cerebellum Controller (Fwd DCN) TO controller_pops.pred_p/n ---
        conn_spec_dcn_f_pred = self.conn_params.dcn_forw_prediction
        w_dcn_pred = conn_spec_dcn_f_pred.weight
        d_dcn_pred = conn_spec_dcn_f_pred.delay
        self.log.debug(
            "Connecting Cerebellum Fwd DCN -> Controller's pred_p/n",
            weight=w_dcn_pred,
            delay=d_dcn_pred,
        )
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_p_view.pop,
            self.controller_pops.pred_p.pop,
            "all_to_all",
            syn_spec={"weight": w_dcn_pred, "delay": d_dcn_pred},
        )
        # DCN minus inhibits Positive Prediction
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_n_view.pop,
            self.controller_pops.pred_p.pop,
            "all_to_all",
            syn_spec={"weight": -w_dcn_pred, "delay": d_dcn_pred},
        )
        # DCN minus drives Negative Prediction
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_n_view.pop,
            self.controller_pops.pred_n.pop,
            "all_to_all",
            syn_spec={"weight": w_dcn_pred, "delay": d_dcn_pred},
        )
        # DCN plus inhibits Negative Prediction
        nest.Connect(
            self.cerebellum.populations.forw_dcnp_p_view.pop,
            self.controller_pops.pred_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_dcn_pred, "delay": d_dcn_pred},
        )

        # --- Connections TO Cerebellum Controller Interfaces (FROM controller_pops) ---
        # MC Out -> Cereb Motor Commands Input
        conn_spec_mc_out_mc = self.conn_params.mc_out_motor_commands
        w_mc_motor = conn_spec_mc_out_mc.weight
        d_mc_motor = conn_spec_mc_out_mc.delay
        self.log.debug(
            "Connecting Controller MC Out -> Cereb Motor Cmds",
            weight=w_mc_motor,
            delay=d_mc_motor,
        )
        nest.Connect(
            self.controller_pops.mc_out_p.pop,
            self.interface_pops.motor_commands_p.pop,
            "all_to_all",
            syn_spec={"weight": w_mc_motor, "delay": d_mc_motor},
        )
        nest.Connect(
            self.controller_pops.mc_out_n.pop,
            self.interface_pops.motor_commands_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_mc_motor, "delay": d_mc_motor},
        )

        # Planner -> Cereb Plan To Inv Input
        conn_spec_plan_pti = self.conn_params.planner_plan_to_inv
        w_plan_inv = conn_spec_plan_pti.weight
        d_plan_inv = conn_spec_plan_pti.delay
        self.log.debug(
            "Connecting Controller Planner -> Cereb PlanToInv",
            weight=w_plan_inv,
            delay=d_plan_inv,
        )
        nest.Connect(
            self.controller_pops.planner_p.pop,
            self.interface_pops.plan_to_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": w_plan_inv, "delay": d_plan_inv},
        )
        nest.Connect(
            self.controller_pops.planner_n.pop,
            self.interface_pops.plan_to_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_plan_inv, "delay": d_plan_inv},
        )

        # Sensory -> Cereb Feedback Input
        conn_spec_sn_fbk_sm = self.conn_params.sn_fbk_smoothed
        w_sn_fbk = conn_spec_sn_fbk_sm.weight
        d_sn_fbk = conn_spec_sn_fbk_sm.delay
        self.log.debug(
            "Connecting Controller Sensory -> Cereb Feedback",
            weight=w_sn_fbk,
            delay=d_sn_fbk,
        )
        nest.Connect(
            self.controller_pops.sn_p.pop,
            self.interface_pops.feedback_p.pop,
            "all_to_all",
            syn_spec={"weight": w_sn_fbk, "delay": d_sn_fbk},
        )
        nest.Connect(
            self.controller_pops.sn_n.pop,
            self.interface_pops.feedback_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_sn_fbk, "delay": d_sn_fbk},
        )

        # Sensory -> Cereb Feedback Inv Input
        conn_spec_sn_fbk_inv = self.conn_params.sn_feedback_inv
        w_sn_finv = conn_spec_sn_fbk_inv.weight
        d_sn_finv = conn_spec_sn_fbk_inv.delay
        self.log.debug(
            "Connecting Controller Sensory -> Cereb FeedbackInv",
            weight=w_sn_finv,
            delay=d_sn_finv,
        )
        nest.Connect(
            self.controller_pops.sn_p.pop,
            self.interface_pops.feedback_inv_p.pop,
            "all_to_all",
            syn_spec={"weight": w_sn_finv, "delay": d_sn_finv},
        )
        nest.Connect(
            self.controller_pops.sn_n.pop,
            self.interface_pops.feedback_inv_n.pop,
            "all_to_all",
            syn_spec={"weight": -w_sn_finv, "delay": d_sn_finv},
        )

        # StateEst -> Cereb State To Inv Input
        # TODO: Check if "planner_plan_to_inv" is the correct conn_spec or if a dedicated one like "state_state_to_inv" is needed.
        conn_spec_state_sti = (
            self.conn_params.planner_plan_to_inv
        )  # Using planner_plan_to_inv as per existing code
        w_state_inv = conn_spec_state_sti.weight
        d_state_inv = conn_spec_state_sti.delay

        self.log.debug(
            "Connecting Controller StateEst -> Cereb StateToInv",
            weight=w_state_inv,
            delay=d_state_inv,
        )
        if self.controller_pops.state_p and self.interface_pops.state_to_inv_p:
            nest.Connect(
                self.controller_pops.state_p.pop,
                self.interface_pops.state_to_inv_p.pop,
                "all_to_all",
                syn_spec={"weight": w_state_inv, "delay": d_state_inv},
            )
        if self.controller_pops.state_n and self.interface_pops.state_to_inv_n:
            nest.Connect(
                self.controller_pops.state_n.pop,
                self.interface_pops.state_to_inv_n.pop,
                "all_to_all",
                syn_spec={"weight": -w_state_inv, "delay": d_state_inv},
            )

        # --- Connections FROM Cerebellum Controller Interfaces (motor_prediction) TO controller_pops.brainstem ---
        conn_spec_mp_bs = self.conn_params.motor_pre_brain_stem
        conn_spec_p_bs = conn_spec_mp_bs.model_dump(exclude_none=True)
        self.log.debug(
            "Connecting Cerebellum motor prediction to Controller brainstem",
            conn_spec=conn_spec_p_bs,
        )
        nest.Connect(
            self.interface_pops.motor_prediction_p.pop,
            self.controller_pops.brainstem_p.pop,
            "all_to_all",
            syn_spec=conn_spec_p_bs,
        )
        conn_spec_n_bs = conn_spec_mp_bs.model_copy(
            update={"weight": -conn_spec_mp_bs.weight}
        ).model_dump(exclude_none=True)
        nest.Connect(
            self.interface_pops.motor_prediction_n.pop,
            self.controller_pops.brainstem_n.pop,
            "all_to_all",
            syn_spec=conn_spec_n_bs,
        )

    # --- Methods to get interface populations (optional, for clarity) ---
    # def get_forward_prediction_outputs( # Obsolete as prediction_p/n are now in Controller
    #     self,
    # ) -> tuple[Optional[PopView], Optional[PopView]]:
    #     """Returns the forward model prediction output PopViews."""
    #     return self.interface_pops.prediction_p, self.interface_pops.prediction_n

    def get_inverse_prediction_outputs(
        self,
    ) -> tuple[Optional[PopView], Optional[PopView]]:
        """Returns the inverse model prediction output PopViews."""
        return (
            self.interface_pops.motor_prediction_p,
            self.interface_pops.motor_prediction_n,
        )

    # Add getters for input interface populations if needed by Controller
    # e.g., get_motor_command_inputs(), get_planner_inputs(), etc.
    # Example:
    # def get_motor_command_inputs(self) -> tuple[Optional[PopView], Optional[PopView]]:
    #     return self.interface_pops.motor_commands_p, self.interface_pops.motor_commands_n
