from typing import Any, Dict, Optional, Tuple

import nest
import numpy as np
import structlog
from CerebellumHandler import CerebellumHandler
from ControllerPopulations import ControllerPopulations
from mpi4py.MPI import Comm

from motorcortex import MotorCortex
from planner import Planner
from population_view import PopView
from stateestimator import StateEstimator_mass

#                       motorcommands.txt
#                               │
#         ┌─────────┐    ┌──────┼──────────────────────────┐
#  t      │         │    │      ▼         Motor Cortex     │    ┌────────────┐
#  r ────▶│ Planner │    │  ┌─────────┐       ┌─────────┐  │───▶│ Smoothing  │
#  a      │(tracking│----│-▶│  Ffwd   │──────▶│   Out   │  │    └────────────┘      __
#  j.txt  │ neuron) │    │  │(tracking│       │  (basic │  │          |     _(\    |@@|
#         └─────────┘    │  │ neuron) │       │  neuron)│  │          ▼    (__/\__ \--/ __
#              │         │  └─────────┘       └─────────┘  │       robotic    \___|----|  |   __
#              │         │                         ▲       │        plant         \ }{ /\ )_ / _\
#              │  +      │  ┌─────────┐            │       │          │           /\__/\ \__O (__
#              └────▶█───┼─▶│   Fbk   │            │       │          │          (--/\--)    \__/
#                    ▲   │  │  (diff  │────────────┘       │          │          _)(  )(_
#                  - │   │  │  neuron)│                    │          ▼         `---''---`
#                    │   │  └─────────┘                    │    ┌────────────────┐
#                    │   └─────────────────────────────────┘    │    Sensory     │
#                    │                                          │     system     │
#                    │                                          └────────────────┘
#            ┌──────────────┐   ┌───────────────┐                      │
#            │    State     │   │  Fbk_smoothed │◀─────────────────────┘
#            │  estimator   │◀──│ (basic neuron)│
#            │(state neuron)│   └───────────────┘
#            └──────────────┘

NJT = 1


class Controller:
    """
    Encapsulates the NEST network components and connections for a single DoF,
    using PopView for recording and a dataclass for population management.
    """

    def __init__(
        self,
        dof_id: int,
        N: int,
        total_time_vect: np.ndarray,
        trajectory_slice: np.ndarray,
        motor_cmd_slice: np.ndarray,
        mc_params: Dict[str, Any],
        plan_params: Dict[str, Any],
        spine_params: Dict[str, Any],
        state_params: Dict[str, Any],
        pops_params: Dict[str, Any],
        conn_params: Dict[str, Any],
        sim_params: Dict[str, Any],
        path_data: str,
        comm: Comm,
        label_prefix: str = "",
        use_cerebellum: bool = False,
        cerebellum_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the controller for one Degree of Freedom.

        Args:
            ...
            use_cerebellum (bool): Flag to enable/disable cerebellum integration.
            cerebellum_config (Optional[Dict[str, Any]]): Configuration for Cerebellum build (paths etc.), required if use_cerebellum is True.
        """
        self.log: structlog.stdlib.BoundLogger = structlog.get_logger(
            f"controller"
        ).bind(controller_dof=dof_id)
        self.log.info("Initializing Controller")
        self.dof_id = dof_id
        self.N = N
        self.total_time_vect = total_time_vect
        self.trajectory_slice = trajectory_slice
        self.motor_cmd_slice = motor_cmd_slice

        # Store parameters (consider dedicated dataclasses per module if very stable)
        self.mc_params = mc_params
        self.plan_params = plan_params
        self.spine_params = spine_params
        self.state_params = state_params
        # self.state_se_params = state_se_params # Store if needed
        self.pops_params = pops_params
        self.conn_params = conn_params
        self.sim_params = sim_params
        self.path_data = path_data
        self.use_cerebellum = use_cerebellum
        self.cerebellum_config = cerebellum_config
        self.comm = comm

        # Use path_data implicitly via PopView labels if saving to file
        # self.label = f"{label_prefix}dof{dof_id}_"
        self.label = f"{label_prefix}"  # Keep label simple as prefix

        self.log.debug(
            "Controller Parameters",
            dof_id=dof_id,
            N=N,
            mc_params=mc_params,
            plan_params=plan_params,
            spine_params=spine_params,
            state_params=state_params,
            pops_params=pops_params,
            conn_params=conn_params,  # Note: Ensure these contain cerebellum sections if needed
            sim_params=sim_params,
            use_cerebellum=self.use_cerebellum,
            cerebellum_config=self.cerebellum_config,
            comm=self.comm,
        )

        # Instantiate the populations dataclass
        self.pops = ControllerPopulations()
        self.cerebellum_handler: Optional[CerebellumHandler] = None

        if use_cerebellum:
            self.cerebellum_handler = self._instantiate_cerebellum_handler(self.pops)
        # this order has a reason to it: beside the obvious dependency connect -> create, there is a more insidious one:
        # Cerebellum ALWAYS tries to import custom_stdp, but if it had been imported already NEST will error out
        # (tracking_neuron_nestml already exists, choose a different model name); this means that Cerebellum must be
        # created first, so that its modules are imported -> custom_stdp exists already and does not need to be loaded again

        # --- Build and Connect ---
        self.log.info("Creating controller blocks...")
        self._create_blocks()
        self.log.info("Connecting internal controller blocks...")
        self._connect_blocks_controller()

        # --- Connect Cerebellum and Controller (needs to be here... >:( )
        if use_cerebellum:
            self.cerebellum_handler.connect_to_main_controller_populations()

        self.log.info("Controller initialization complete.")

    def _instantiate_cerebellum_handler(self, controller_pops: ControllerPopulations):
        """Instantiates the internal CerebellumHandler."""
        self.log.info("Instantiating internal CerebellumHandler")
        if self.cerebellum_config is None:
            raise ValueError(
                "Cerebellum config must be provided when use_cerebellum is True"
            )

        cereb_pop_keys = [
            "prediction",
            "feedback",
            "motor_commands",
            "error",
            "plan_to_inv",
            "state_to_inv",
            "error_i",
            "motor_pred",
            "feedback_inv",
        ]
        cereb_conn_keys = [
            "dcn_forw_prediction",
            "error_io_f",
            "dcn_f_error",  # Fwd connections
            "feedback_error",  # Used by CerebellumHandler for fwd error calculation
            "plan_to_inv_error_inv",
            "state_error_inv",  # Inv error connections (state_error_inv might not exist, handled below)
            "error_inv_io_i",
            "dcn_i_motor_pred",  # Inv connections
            # "sn_feedback_inv" is used by controller.py to connect TO cerebellum_controller, not internally by it.
            # Connections *from* SDC *to* CerebController interfaces are handled in SDC._connect_blocks
            # Connections *from* CerebController *to* SDC interfaces are handled in SDC._connect_blocks
        ]

        # cereb_pops_params = {k: self.pops_params[k] for k in cereb_pop_keys}
        # cereb_conn_params = {k: self.conn_params[k] for k in cereb_conn_keys}
        # TODO different parameter sets could be nice :)
        cereb_pops_params = self.pops_params
        cereb_conn_params = self.conn_params

        try:
            cerebellum_controller = CerebellumHandler(
                N=self.N,
                total_time_vect=self.total_time_vect,
                sim_params=self.sim_params,
                pops_params=cereb_pops_params,
                conn_params=cereb_conn_params,
                cerebellum_config=self.cerebellum_config,
                path_data=self.path_data,
                label_prefix=f"{self.label}cereb_",
                dof_id=self.dof_id,
                comm=self.comm,
                controller_pops=controller_pops,
            )
            self.log.info("Internal CerebellumHandler instantiated successfully.")
            return cerebellum_controller
        except Exception as e:
            self.log.error(
                "Failed to instantiate internal CerebellumHandler",
                error=str(e),
                exc_info=True,
            )
            raise

    # --- 1. Block Creation ---
    def _create_blocks(self):
        """Creates all neuron populations using PopView for this DoF."""
        self.log.debug("Building planner block")
        self._build_planner(to_file=True)
        self.log.debug("Building motor cortex block")
        self._build_motor_cortex(to_file=True)
        self.log.debug("Building state estimator block")
        self._build_state_estimator(to_file=True)
        self.log.debug("Building sensory neurons block")
        self._build_sensory_neurons(to_file=True)
        self.log.debug(
            "Building prediction neurons block"
        )  # These are the scaling neurons for cerebellar fwd output or other prediction source
        self._build_prediction_neurons(to_file=True)
        self.log.debug("Building feedback smoothed neurons block")
        self._build_fbk_smoothed_neurons(to_file=True)
        self.log.debug("Building brainstem block")
        self._build_brainstem(to_file=True)

    # --- Helper for PopView Creation ---
    def _create_pop_view(
        self, nest_pop: nest.NodeCollection, base_label: str, to_file: bool
    ) -> PopView:
        """Creates a PopView instance with appropriate label."""
        full_label = f"{self.label}{base_label}" if to_file else ""
        return PopView(
            nest_pop, self.total_time_vect, to_file=to_file, label=full_label
        )

    # --- Build Methods (Example: Planner) ---
    def _build_planner(self, to_file=False):
        # --- Replicate Planner logic or use Planner class ---
        p_params = self.plan_params
        pop_params = self.pops_params  # Assuming params structure
        N = self.N
        self.log.debug(
            "Initializing Planner sub-module",
            N=N,
            njt=NJT,
            kpl=p_params["kpl"],
            base_rate=p_params["base_rate"],
            kp=p_params["kp"],
        )
        self.planner = Planner(
            N,
            NJT,
            self.total_time_vect,
            self.trajectory_slice,
            self.path_data,
            p_params["kpl"],
            p_params["base_rate"],
            p_params["kp"],
        )
        self.pops.planner_p = self.planner.pops_p[0]
        self.pops.planner_n = self.planner.pops_n[0]

    def _build_motor_cortex(self, to_file=False):
        self.log.debug(
            "Initializing MotorCortex sub-module",
            N=self.N,
            njt=1,
            mc_params=self.mc_params,
        )
        self.mc = MotorCortex(
            self.N, NJT, self.total_time_vect, self.motor_cmd_slice, **self.mc_params
        )
        self.pops.mc_ffwd_p = self.mc.ffwd_p[0]
        self.pops.mc_ffwd_n = self.mc.ffwd_n[0]
        self.pops.mc_fbk_p = self.mc.fbk_p[0]
        self.pops.mc_fbk_n = self.mc.fbk_n[0]
        self.pops.mc_out_p = self.mc.out_p[0]
        self.pops.mc_out_n = self.mc.out_n[0]

    def _build_state_estimator(self, to_file=False):
        buf_sz = self.state_params["buffer_size"]
        N = self.N

        additional_state_params = {
            "N_fbk": N,
            "N_pred": N,
            "fbk_bf_size": N * int(buf_sz / self.sim_params["res"]),
            "pred_bf_size": N * int(buf_sz / self.sim_params["res"]),
            "time_wait": self.sim_params["timeWait"],
            "time_trial": self.sim_params["timeMax"] + self.sim_params["timeWait"],
        }
        self.state_params.update(additional_state_params)
        self.log.debug(
            "Initializing StateEstimator_mass",
            N=N,
            njt=NJT,
            state_params=self.state_params,
        )
        self.stEst = StateEstimator_mass(
            N, NJT, self.total_time_vect, **self.state_params
        )
        self.pops.state_p = self.stEst.pops_p[0]
        self.pops.state_n = self.stEst.pops_n[0]

    def _build_sensory_neurons(self, to_file=False):
        """Parrot neurons for sensory feedback input"""
        pop_p = nest.Create("parrot_neuron", self.N)
        self.pops.sn_p = self._create_pop_view(pop_p, "sensoryneur_p", to_file)
        pop_n = nest.Create("parrot_neuron", self.N)
        self.pops.sn_n = self._create_pop_view(pop_n, "sensoryneur_n", to_file)

    def _build_prediction_neurons(self, to_file=False):
        """
        Builds internal prediction neurons (diff_neuron_nestml).
        These neurons always exist to 'scale' the prediction signal before it goes to the State Estimator.
        If cerebellum is used, the prediction signal comes from cerebellum_controller.forw_DCNp_plus/minus,
        which then connects to these pred_p/n neurons.
        """
        self.log.debug("Building prediction scaling neurons (pred_p, pred_n)")
        params = self.pops_params["prediction"]
        pop_params = {
            "kp": params["kp"],
            "buffer_size": params["buffer_size"],
            "base_rate": params["base_rate"],
            "simulation_steps": len(self.total_time_vect),
        }

        pop_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(pop_p, {**pop_params, "pos": True})
        self.pops.pred_p = self._create_pop_view(pop_p, "pred_p", to_file)

        pop_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(pop_n, {**pop_params, "pos": False})
        self.pops.pred_n = self._create_pop_view(pop_n, "pred_n", to_file)

    def _build_fbk_smoothed_neurons(self, to_file=False):
        """Neurons for smoothing feedback"""
        params = self.pops_params["fbk_smoothed"]
        pop_params = {
            "kp": params["kp"],
            "buffer_size": params["buffer_size"],
            "base_rate": params["base_rate"],
            "simulation_steps": len(self.total_time_vect),
        }
        self.log.debug("Creating feedback neurons", **pop_params)

        pop_p = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(pop_p, {**pop_params, "pos": True})
        self.pops.fbk_smooth_p = self._create_pop_view(pop_p, "fbk_smooth_p", to_file)

        pop_n = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(pop_n, {**pop_params, "pos": False})
        self.pops.fbk_smooth_n = self._create_pop_view(pop_n, "fbk_smooth_n", to_file)

    def _build_brainstem(self, to_file=False):
        """Basic neurons for output stage"""
        params = self.pops_params["brain_stem"]
        pop_params = {
            "kp": params["kp"],
            "buffer_size": params["buffer_size"],
            "base_rate": params["base_rate"],
            "simulation_steps": len(self.total_time_vect),
        }
        self.log.debug("Creating output neurons (brainstem)", **pop_params)

        pop_p = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(pop_p, {**pop_params, "pos": True})
        self.pops.brainstem_p = self._create_pop_view(pop_p, "brainstem_p", to_file)

        pop_n = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(pop_n, {**pop_params, "pos": False})
        self.pops.brainstem_n = self._create_pop_view(pop_n, "brainstem_n", to_file)

    # --- 2. Block Connection ---
    def _connect_blocks_controller(self):
        """Connects the created populations using PopView attributes."""
        self.log.debug("Connecting internal controller blocks")

        # Planner -> Motor Cortex Feedback Input
        # if self.pops.planner_p and self.pops.mc_fbk_p:  # Check populations exist
        w = self.conn_params["planner_mc_fbk"]["weight"]
        d = self.conn_params["planner_mc_fbk"]["delay"]
        self.log.debug("Connecting Planner to MC Fbk", weight=w, delay=d)
        nest.Connect(
            self.pops.planner_p.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec={"weight": w, "delay": d},
        )
        nest.Connect(
            self.pops.planner_p.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec={"weight": w, "delay": d},
        )
        nest.Connect(
            self.pops.planner_n.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec={"weight": -w, "delay": d},
        )
        nest.Connect(
            self.pops.planner_n.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec={"weight": -w, "delay": d},
        )

        # State Estimator -> Motor Cortex Feedback Input (Inhibitory)
        # if self.pops.state_p and self.pops.mc_fbk_p:
        conn_spec = self.conn_params["state_mc_fbk"]
        self.log.debug(
            "Connecting StateEst to MC Fbk (Inhibitory)", conn_spec=conn_spec
        )
        nest.Connect(
            self.pops.state_p.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec=conn_spec,
        )
        nest.Connect(
            self.pops.state_p.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec=conn_spec,
        )
        conn_spec["weight"] = -conn_spec["weight"]
        nest.Connect(
            self.pops.state_n.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec=conn_spec,
        )
        nest.Connect(
            self.pops.state_n.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec=conn_spec,
        )

        # Motor Cortex Output -> Brainstem
        conn_spec = self.conn_params["mc_out_brain_stem"]
        self.log.debug("Connecting MC out to brainstem", conn_spec=conn_spec)
        nest.Connect(
            self.pops.mc_out_p.pop,
            self.pops.brainstem_p.pop,
            "all_to_all",
            syn_spec=conn_spec,
        )
        # TODO: this is just sick. who on earth would ever think
        # that a secret minus is better than just putting it in the weight definition??????
        conn_spec["weight"] = -conn_spec["weight"]
        nest.Connect(
            self.pops.mc_out_n.pop,
            self.pops.brainstem_n.pop,
            "all_to_all",
            syn_spec=conn_spec,
        )

        # Sensory Input -> Feedback Smoothed Neurons
        conn_spec = self.conn_params["sn_fbk_smoothed"]
        self.log.debug("Connecting sensory to smoothing", conn_spec=conn_spec)
        nest.Connect(
            self.pops.sn_p.pop,
            self.pops.fbk_smooth_p.pop,
            "all_to_all",
            syn_spec=conn_spec,
        )
        conn_spec["weight"] = -conn_spec["weight"]
        nest.Connect(
            self.pops.sn_n.pop,
            self.pops.fbk_smooth_n.pop,
            "all_to_all",
            syn_spec=conn_spec,
        )

        # Connections INTO State Estimator (Using receptor types)
        st_p = self.pops.state_p.pop
        st_n = self.pops.state_n.pop

        w_fbk_sm = self.conn_params["fbk_smoothed_state"]["weight"]
        self.log.debug("Connecting smoothed sensory to state", weight=w_fbk_sm)
        for i, pre in enumerate(self.pops.fbk_smooth_p.pop):
            nest.Connect(
                pre,
                st_p,
                "all_to_all",
                syn_spec={"weight": w_fbk_sm, "receptor_type": i + 1},
            )
        for i, pre in enumerate(self.pops.fbk_smooth_n.pop):
            nest.Connect(
                pre,
                st_n,
                "all_to_all",
                syn_spec={"weight": w_fbk_sm, "receptor_type": i + 1},
            )
        # Prediction (self.pops.pred_p/n) -> State Estimator (Receptors N+1 to 2N)
        # These connections are always made, as pred_p/n always exist.
        offset = self.N + 1  # Start receptor types after the first N for sensory
        weight = self.conn_params["pred_state"]["weight"]
        self.log.debug(
            "Connecting self.pops.pred_p/n to state estimator",
            weight=weight,
            receptor_offset=offset,
        )
        for i, pre in enumerate(self.pops.pred_p.pop):
            nest.Connect(
                pre,
                st_p,
                "all_to_all",
                syn_spec={"weight": weight, "receptor_type": i + offset},
            )
        for i, pre in enumerate(self.pops.pred_n.pop):
            nest.Connect(
                pre,
                st_n,
                "all_to_all",
                syn_spec={"weight": weight, "receptor_type": i + offset},
            )
        # Note: The MC Output -> Brainstem connection happens in both cases and is handled above

    # --- 3. Input/Output Access ---
    # These return PopView objects now, callers can access .pop if needed
    def get_sensory_input_popviews(self) -> Tuple[Optional[PopView], Optional[PopView]]:
        """Returns the positive and negative sensory input PopViews."""
        return self.pops.sn_p, self.pops.sn_n

    def get_brainstem_output_popviews(
        self,
    ) -> Tuple[Optional[PopView], Optional[PopView]]:
        """Returns the positive and negative brainstem output PopViews."""
        return self.pops.brainstem_p, self.pops.brainstem_n
