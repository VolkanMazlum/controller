# complete_control/CerebellumInterfacePopulations.py
from dataclasses import dataclass, field
from typing import List, Optional

import nest

# Assuming PopView is imported from population_view.py
from population_view import PopView


@dataclass
class CerebellumInterfacePopulations:
    """Holds the PopView instances for the interface populations of the CerebellumController."""

    # --- Inputs TO CerebellumController ---
    # (Populations created in SingleDOFController, connected TO these interface pops)

    # From Motor Cortex Output (via motor_commands scaling neurons)
    motor_commands_p: Optional[PopView] = None
    motor_commands_n: Optional[PopView] = None

    # From Planner (via plan_to_inv scaling neurons)
    plan_to_inv_p: Optional[PopView] = None
    plan_to_inv_n: Optional[PopView] = None

    # From Sensory Neurons (via feedback scaling neurons)
    feedback_p: Optional[PopView] = None
    feedback_n: Optional[PopView] = None

    # From Sensory Neurons (via feedback_inv scaling neurons)
    feedback_inv_p: Optional[PopView] = None
    feedback_inv_n: Optional[PopView] = None

    # From State Estimator (via state_to_inv scaling neurons)
    state_to_inv_p: Optional[PopView] = None
    state_to_inv_n: Optional[PopView] = None

    # Forward Model Error Calculation (Input to Fwd IO, calculated from feedback and fwd_dcn)
    error_p: Optional[PopView] = None
    error_n: Optional[PopView] = None

    # --- Outputs FROM CerebellumController ---
    # (Populations created within CerebellumController, connected FROM these interface pops)

    # Inverse Model Prediction (Output from DCN_inv, scaled by motor_prediction neurons)
    motor_prediction_p: Optional[PopView] = None
    motor_prediction_n: Optional[PopView] = None

    # Inverse Model Error Calculation (Input to Inv IO, calculated from plan_to_inv and state_to_inv)
    error_inv_p: Optional[PopView] = None
    error_inv_n: Optional[PopView] = None

    # Helper to get all valid PopView objects
    def get_all_views(self) -> List[PopView]:
        views = []
        for pop_field in self.__dataclass_fields__:
            view = getattr(self, pop_field)
            if isinstance(view, PopView):
                views.append(view)
        return views
