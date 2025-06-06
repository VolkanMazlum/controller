from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class SingleSynapseParams(BaseModel):
    model_config: ClassVar = {"frozen": True}
    weight: float
    delay: Optional[float] = None
    receptor_type: Optional[int] = None


class ConnectionsParams(BaseModel):
    model_config: ClassVar = {"frozen": True}
    dcn_forw_prediction: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.3,
            delay=0.1,
        )
    )
    sn_fbk_smoothed: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.028,
            delay=0.1,
        )
    )
    pred_state: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            receptor_type=1,
        )
    )
    fbk_smoothed_state: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.10,
            receptor_type=2,
        )
    )
    sn_state: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.6317663917438847,
            receptor_type=2,
        )
    )
    planner_mc_fbk: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            delay=0.1,
        )
    )
    state_mc_fbk: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=-0.95,
            delay=0.1,
        )
    )
    mc_out_motor_commands: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.03,
            delay=0.1,
        )
    )
    motor_commands_mossy_forw: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            delay=0.1,
        )
    )
    sn_feedback: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.001,
            delay=0.1,
        )
    )
    dcn_f_error: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.1,
            delay=0.1,
        )
    )
    feedback_error: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.1,
            delay=0.1,
        )
    )
    error_io_f: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=3.0,
            delay=0.1,
            receptor_type=1,
        )
    )
    planner_plan_to_inv: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.001,
            delay=0.1,
        )
    )
    plan_to_inv_mossy_i: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            delay=0.1,
        )
    )
    dcn_i_motor_pred: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.5,
            delay=0.1,
        )
    )
    motor_pred_mc_out: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.1,
            delay=0.1,
        )
    )
    motor_pre_brain_stem: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.005,
            delay=0.1,
        )
    )
    mc_out_brain_stem: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.1,
            delay=0.1,
        )
    )
    sn_feedback_inv: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.001,
            delay=0.1,
        )
    )
    feedback_inv_error_inv: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            delay=0.1,
        )
    )
    state_error_inv: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=0.5,
            delay=0.1,
        )
    )
    plan_to_inv_error_inv: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=1.0,
            delay=0.1,
        )
    )
    error_inv_io_i: SingleSynapseParams = Field(
        default_factory=lambda: SingleSynapseParams(
            weight=3.0,
            delay=0.1,
            receptor_type=1,
        )
    )
