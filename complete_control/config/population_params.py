from typing import ClassVar, Dict

from pydantic import BaseModel, Field


class SinglePopParams(BaseModel):
    model_config: ClassVar = {"frozen": True}
    kp: float
    buffer_size: float
    base_rate: float


class PopulationsParams(BaseModel):
    model_config: ClassVar = {"frozen": True}
    prediction: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(
            kp=4.0, buffer_size=20.0, base_rate=50.0
        )
    )
    motor_commands: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(kp=0.1, buffer_size=25.0, base_rate=0.0)
    )
    brain_stem: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(kp=0.2, buffer_size=10.0, base_rate=0.0)
    )
    feedback: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(kp=1.0, buffer_size=10.0, base_rate=0.0)
    )
    fbk_smoothed: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(
            kp=1.0, buffer_size=25.0, base_rate=100.0
        )
    )
    error: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(
            kp=1.0, buffer_size=30.0, base_rate=-20.0
        )
    )
    plan_to_inv: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(kp=1.0, buffer_size=10.0, base_rate=0.0)
    )
    motor_pred: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(
            kp=0.05, buffer_size=20.0, base_rate=40.0
        )
    )
    feedback_inv: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(kp=1.0, buffer_size=10.0, base_rate=0.0)
    )
    error_i: SinglePopParams = Field(
        default_factory=lambda: SinglePopParams(
            kp=1.0, buffer_size=30.0, base_rate=-20.0
        )
    )
