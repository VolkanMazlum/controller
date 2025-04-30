# controller.py (Updated SingleDOFController)

from typing import Any, Dict, Optional

import nest
import numpy as np
from ControllerPopulations import ControllerPopulations

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


class SingleDOFController:
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
        state_se_params: Dict[
            str, Any
        ],  # Note: state_se_params wasn't used before, check necessity
        pops_params: Dict[str, Any],
        conn_params: Dict[str, Any],
        sim_params: Dict[str, Any],
        path_data: str,  # Used implicitly by PopView if to_file=True
        label_prefix: str = "",
    ):
        """
        Initializes the controller for one Degree of Freedom.
        (Args documentation mostly unchanged)
        """
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

        # Use path_data implicitly via PopView labels if saving to file
        # self.label = f"{label_prefix}dof{dof_id}_"
        self.label = f"{label_prefix}"

        # Instantiate the populations dataclass
        self.pops = ControllerPopulations()

        # --- Build and Connect ---
        # Pass `to_file=True` to PopView constructor where needed
        self._create_blocks()
        self._connect_blocks()

    # --- 1. Block Creation ---
    def _create_blocks(self):
        """Creates all neuron populations using PopView for this DoF."""
        # Note: PopView internally creates the spike detector.
        # Pass to_file=True and label to PopView if you want NEST ASCII output.
        self._build_planner(to_file=True)
        self._build_motor_cortex(to_file=True)
        self._build_state_estimator(to_file=True)
        self._build_sensory_neurons(to_file=True)
        self._build_prediction_neurons(to_file=True)
        self._build_fbk_smoothed_neurons(to_file=True)
        self._build_brainstem(to_file=True)
        # No _create_recorders needed

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
        N = 50
        njt = 1
        # TODO: fix this
        print("--- Refactored Controller Planner Init ---")
        # N and njt are hardcoded to 50 and 1 inside this method currently
        print(f"N (hardcoded): {N}")
        print(f"njt (hardcoded): {njt}")
        print(
            f"total_time_vect len: {len(self.total_time_vect)}, start: {self.total_time_vect[0]:.2f}, end: {self.total_time_vect[-1]:.2f}, res: {self.sim_params['res']}"
        )
        print(
            f"trajectory_slice shape: {self.trajectory_slice.shape}, start: {self.trajectory_slice[0]:.4f}, end: {self.trajectory_slice[-1]:.4f}"
        )
        print(f"path_data: {self.path_data}")
        print(f"plan_params['kpl']: {p_params['kpl']}")
        print(f"plan_params['base_rate']: {p_params['base_rate']}")
        print(f"plan_params['kp']: {p_params['kp']}")
        self.planner = Planner(
            N,
            njt,
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
        # --- Replicate MC logic or use MotorCortex class ---
        # TODO: fix this
        N = 50
        njt = 1
        print("--- Refactored Controller MotorCortex Init ---")
        # N and njt are hardcoded to 50 and 1 inside this method currently
        print(f"N (hardcoded): {N}")
        print(f"njt (hardcoded): {njt}")
        print(
            f"total_time_vect len: {len(self.total_time_vect)}, start: {self.total_time_vect[0]:.2f}, end: {self.total_time_vect[-1]:.2f}, res: {self.sim_params['res']}"
        )
        print(
            f"motor_cmd_slice shape: {self.motor_cmd_slice.shape}, start: {self.motor_cmd_slice[0]:.4f}, end: {self.motor_cmd_slice[-1]:.4f}"
        )
        print(f"mc_params: {self.mc_params}")
        self.mc = MotorCortex(
            N, njt, self.total_time_vect, self.motor_cmd_slice, **self.mc_params
        )
        self.pops.mc_ffwd_p = self.mc.ffwd_p[0]
        self.pops.mc_ffwd_n = self.mc.ffwd_n[0]
        self.pops.mc_fbk_p = self.mc.fbk_p[0]
        self.pops.mc_fbk_n = self.mc.fbk_n[0]
        self.pops.mc_out_p = self.mc.out_p[0]
        self.pops.mc_out_n = self.mc.out_n[0]
        """
        # Placeholder using basic neurons for structure:
        # FFWD
        ffwd_p = nest.Create("parrot_neuron", self.N)  # Placeholder type
        # TODO: Configure ffwd_p (e.g., stimulus from motor_cmd_slice)
        self.pops.mc_ffwd_p = self._create_pop_view(ffwd_p, "mc_ffwd_p", to_file)
        ffwd_n = nest.Create("parrot_neuron", self.N)  # Placeholder type
        # TODO: Configure ffwd_n
        self.pops.mc_ffwd_n = self._create_pop_view(ffwd_n, "mc_ffwd_n", to_file)

        # FBK Input Neurons (likely simple neurons integrating Planner and StateEst)
        fbk_p = nest.Create("iaf_cond_alpha", self.N)  # Placeholder type
        # TODO: Configure fbk_p (params from mc_params)
        self.pops.mc_fbk_p = self._create_pop_view(fbk_p, "mc_fbk_p", to_file)
        fbk_n = nest.Create("iaf_cond_alpha", self.N)  # Placeholder type
        # TODO: Configure fbk_n
        self.pops.mc_fbk_n = self._create_pop_view(fbk_n, "mc_fbk_n", to_file)

        # OUT Neurons (integrating FFWD and FBK)
        out_p = nest.Create("iaf_cond_alpha", self.N)  # Placeholder type
        # TODO: Configure out_p
        self.pops.mc_out_p = self._create_pop_view(out_p, "mc_out_p", to_file)
        out_n = nest.Create("iaf_cond_alpha", self.N)  # Placeholder type
        # TODO: Configure out_n
        self.pops.mc_out_n = self._create_pop_view(out_n, "mc_out_n", to_file)

        # TODO: Add internal MC connections (ffwd+fbk -> out) later in _connect_blocks if not handled by MotorCortex class
        """

    def _build_state_estimator(self, to_file=False):
        # --- Replicate StateEstimator logic or use StateEstimator_mass class ---
        # Requires careful setup of receptor types if not using the class
        num_receptors = 2 * self.N  # Example: N for fbk, N for pred
        buf_sz = self.state_params["buffer_size"]
        # Use self.N passed during initialization consistently
        N = self.N  # Use instance N
        njt = 1  # Assuming always 1 for SingleDOFController

        additional_state_params = {
            "N_fbk": N,  # Use instance N
            "N_pred": N,  # Use instance N
            "fbk_bf_size": N * int(buf_sz / self.sim_params["res"]),  # Use instance N
            "pred_bf_size": N * int(buf_sz / self.sim_params["res"]),  # Use instance N
            "time_wait": self.sim_params["timeWait"],
            "time_trial": self.sim_params["timeMax"] + self.sim_params["timeWait"],
        }
        self.state_params.update(additional_state_params)
        print("--- Refactored Controller StateEstimator Init ---")
        print(f"N: {N}")  # Note: This N is self.N assigned at line 167
        print(f"njt: {njt}")  # Note: This njt is hardcoded to 1 at line 168
        print(
            f"total_time_vect len: {len(self.total_time_vect)}, start: {self.total_time_vect[0]:.2f}, end: {self.total_time_vect[-1]:.2f}, res: {self.sim_params['res']}"
        )
        print(
            f"state_params for StateEstimator_mass: {self.state_params}"
        )  # Note: This is self.state_params after update at line 178
        self.stEst = StateEstimator_mass(
            N, njt, self.total_time_vect, **self.state_params  # Use instance N
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
        """Diff neurons for prediction scaling"""
        params = self.pops_params["prediction"]
        pop_p = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(
            pop_p,
            {
                "kp": params["kp"],
                "pos": True,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.pred_p = self._create_pop_view(pop_p, "pred_p", to_file)

        pop_n = nest.Create("diff_neuron_nestml", self.N)
        nest.SetStatus(
            pop_n,
            {
                "kp": params["kp"],
                "pos": False,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.pred_n = self._create_pop_view(pop_n, "pred_n", to_file)
        # TODO: Add input connection (e.g., from cerebellum) if needed

    def _build_fbk_smoothed_neurons(self, to_file=False):
        """Neurons for smoothing feedback"""
        params = self.pops_params["fbk_smoothed"]
        pop_p = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(
            pop_p,
            {
                "kp": params["kp"],
                "pos": True,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.fbk_smooth_p = self._create_pop_view(pop_p, "fbk_smooth_p", to_file)
        print("\n\nCreated population view for fbk_smooth_p")

        pop_n = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(
            pop_n,
            {
                "kp": params["kp"],
                "pos": False,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.fbk_smooth_n = self._create_pop_view(pop_n, "fbk_smooth_n", to_file)

    def _build_brainstem(self, to_file=False):
        """Basic neurons for output stage"""
        params = self.pops_params["brain_stem"]
        pop_p = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(
            pop_p,
            {
                "kp": params["kp"],
                "pos": True,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.brainstem_p = self._create_pop_view(pop_p, "brainstem_p", to_file)

        pop_n = nest.Create("basic_neuron_nestml", self.N)
        nest.SetStatus(
            pop_n,
            {
                "kp": params["kp"],
                "pos": False,
                "buffer_size": params["buffer_size"],
                "base_rate": params["base_rate"],
                "simulation_steps": len(self.total_time_vect),
            },
        )
        self.pops.brainstem_n = self._create_pop_view(pop_n, "brainstem_n", to_file)

    # --- 2. Block Connection ---
    def _connect_blocks(self):
        """Connects the created populations using PopView attributes."""
        res = self.sim_params["res"]

        # Planner -> Motor Cortex Feedback Input
        # if self.pops.planner_p and self.pops.mc_fbk_p:  # Check populations exist
        w = self.conn_params["planner_mc_fbk"]["weight"]
        d = self.conn_params["planner_mc_fbk"]["delay"]
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
        w = self.conn_params["state_mc_fbk"]["weight"]
        nest.Connect(
            self.pops.state_p.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec={"weight": w, "delay": res},
        )
        nest.Connect(
            self.pops.state_p.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec={"weight": w, "delay": res},
        )
        nest.Connect(
            self.pops.state_n.pop,
            self.pops.mc_fbk_p.pop,
            "one_to_one",
            syn_spec={"weight": -w, "delay": res},
        )
        nest.Connect(
            self.pops.state_n.pop,
            self.pops.mc_fbk_n.pop,
            "one_to_one",
            syn_spec={"weight": -w, "delay": res},
        )

        # Motor Cortex Output -> Brainstem
        conn_spec = self.conn_params["mc_out_brain_stem"]
        nest.Connect(
            self.pops.mc_out_p.pop,
            self.pops.brainstem_p.pop,
            "all_to_all",
            syn_spec={"weight": conn_spec["weight"], "delay": conn_spec["delay"]},
        )
        nest.Connect(
            self.pops.mc_out_n.pop,
            self.pops.brainstem_n.pop,
            "all_to_all",
            # TODO: this is just sick. who on earth would ever think
            # that a secret minus is better than just putting it in the weight definition??????
            syn_spec={"weight": -conn_spec["weight"], "delay": conn_spec["delay"]},
        )

        # Sensory Input -> Feedback Smoothed Neurons
        conn_spec = self.conn_params["sn_fbk_smoothed"]
        nest.Connect(
            self.pops.sn_p.pop,
            self.pops.fbk_smooth_p.pop,
            "all_to_all",
            syn_spec={"weight": conn_spec["weight"], "delay": conn_spec["delay"]},
        )
        nest.Connect(
            self.pops.sn_n.pop,
            self.pops.fbk_smooth_n.pop,
            "all_to_all",
            syn_spec={"weight": -conn_spec["weight"], "delay": conn_spec["delay"]},
        )

        # Connections INTO State Estimator (Using receptor types)
        st_p = self.pops.state_p.pop
        st_n = self.pops.state_n.pop

        # Option 1: Feedback Smoothed -> State Estimator (Receptors 1 to N)
        w_fbk_sm = self.conn_params["fbk_smoothed_state"]["weight"]
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
        # in original script, joints that are not the controlled joints are not smoothed. not exactly sure why.
        # w_sn_st = self.conn_params["sn_state"]["weight"]
        # for i, pre in enumerate(self.pops.sn_p.pop):
        #     nest.Connect(
        #         pre,
        #         st_p,
        #         "all_to_all",
        #         syn_spec={"weight": w_sn_st, "receptor_type": i + 1},
        #     )
        # for i, pre in enumerate(self.pops.sn_n.pop):
        #     nest.Connect(
        #         pre,
        #         st_n,
        #         "all_to_all",
        #         syn_spec={"weight": w_sn_st, "receptor_type": i + 1},
        #     )

        # Prediction -> State Estimator (Receptors N+1 to 2N)
        receptor_offset = self.N + 1
        for i, pre in enumerate(self.pops.pred_p.pop):
            nest.Connect(
                pre,
                st_p,
                "all_to_all",
                syn_spec={"weight": 1.0, "receptor_type": i + receptor_offset},
            )
        for i, pre in enumerate(self.pops.pred_n.pop):
            nest.Connect(
                pre,
                st_n,
                "all_to_all",
                syn_spec={"weight": 1.0, "receptor_type": i + receptor_offset},
            )

    # --- 3. Input/Output Access ---
    # These return PopView objects now, callers can access .pop if needed
    def get_sensory_input_popviews(self) -> tuple[Optional[PopView], Optional[PopView]]:
        """Returns the positive and negative sensory input PopViews."""
        return self.pops.sn_p, self.pops.sn_n

    def get_brainstem_output_popviews(
        self,
    ) -> tuple[Optional[PopView], Optional[PopView]]:
        """Returns the positive and negative brainstem output PopViews."""
        return self.pops.brainstem_p, self.pops.brainstem_n

    # No get_recorder needed. Access data via PopView methods (e.g., get_events, computePSTH)
    # Direct access: controller.pops.planner_p.get_events()

    def connect_external_prediction(
        self, pred_pop_p: nest.NodeCollection, pred_pop_n: nest.NodeCollection
    ):
        """Connects external prediction populations (e.g., shared cerebellum)
        to this controller's prediction scaling neurons."""
        if self.pops.pred_p:
            # Connect to the NEST population *inside* the PopView
            nest.Connect(
                pred_pop_p, self.pops.pred_p.pop, "all_to_all", syn_spec={"weight": 1.0}
            )
            nest.Connect(
                pred_pop_n, self.pops.pred_n.pop, "all_to_all", syn_spec={"weight": 1.0}
            )
