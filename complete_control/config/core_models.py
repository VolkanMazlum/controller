from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel, Field, FilePath, computed_field
from utils_common.git_utils import get_git_commit_hash

from . import paths


class MetaInfo(BaseModel, frozen=True):
    controller_commit_hash: str = Field(
        default_factory=lambda: get_git_commit_hash(paths.ROOT)
    )
    cerebellum_commit_hash: str = Field(
        default_factory=lambda: get_git_commit_hash(paths.CEREBELLUM)
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    run_id: str  # This will be set at runtime


class RobotSpecParams(BaseModel, frozen=True):
    mass: List[float] = [1.89]
    links: List[float] = [0.31]
    I: List[float] = [0.00189]

    model_config = {
        "arbitrary_types_allowed": True  # For potential np.ndarray use later
    }


class ExperimentParams(BaseModel, frozen=True):
    init_joint_angle: float = 75
    tgt_joint_angle: float = 15
    robot_spec: RobotSpecParams = Field(default_factory=lambda: RobotSpecParams())
    # frcFld_angle: float  # unused for now
    # frcFld_k: float  # unused for now
    # ff_application: float  # unused for now
    # ff_removal: float  # unused for now
    # cerebellum_application_forw: float # unused for now
    # cerebellum_application_inv: float # unused for now

    @computed_field
    @property
    def init_pos_angle_rad(self) -> float:
        return np.deg2rad(self.init_joint_angle)

    @computed_field
    @property
    def tgt_pos_angle_rad(self) -> float:
        return np.deg2rad(self.tgt_joint_angle)


class SimulationParams(BaseModel, frozen=True):
    resolution: float = 0.1  # ms
    time_prep: float = 150.0  # ms
    time_move: float = 300.0  # ms
    time_post: float = 50.0  # ms
    n_trials: int = 1

    seed: int = 12345

    @computed_field
    @property
    def duration_single_trial_ms(self) -> float:
        return self.time_prep + self.time_move + self.time_post

    @computed_field
    @property
    def total_duration_all_trials_ms(self) -> float:
        return self.duration_single_trial_ms * self.n_trials

    @classmethod
    def get_default(cls, field_name: str):
        """Get default value for any field"""
        return cls.model_fields[field_name].default

    @classmethod
    def get_default_seed(cls):
        return cls.model_fields["seed"].default


class BrainParams(BaseModel, frozen=True):
    population_size: int = 50
    first_id_sens_neurons: int = 0  # not sure why we need this.


class MusicParams(BaseModel, frozen=True):
    const: float = 1e-6  # Constant to subtract to avoid rounding errors (ms)
    input_latency: float = 0.0001  # seconds
