from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from .bsb_models import BSBConfigCopies, BSBConfigPaths
from .connection_params import ConnectionsParams
from .core_models import (
    BrainParams,
    ExperimentParams,
    MetaInfo,
    MusicParams,
    SimulationParams,
)
from .module_params import ModuleContainerConfig
from .population_params import PopulationsParams


class MasterParams(BaseModel):
    model_config: ClassVar = {
        "frozen": True,
        "arbitrary_types_allowed": True,
    }
    meta: MetaInfo
    NJT: int = 1
    simulation: SimulationParams = Field(default_factory=lambda: SimulationParams())
    experiment: ExperimentParams = Field(default_factory=lambda: ExperimentParams())
    brain: BrainParams = Field(default_factory=lambda: BrainParams())
    music: MusicParams = Field(default_factory=lambda: MusicParams())
    bsb_config_paths: BSBConfigPaths = Field(default_factory=lambda: BSBConfigPaths())
    bsb_config_copies: BSBConfigCopies = Field(
        default_factory=lambda: BSBConfigCopies()
    )

    modules: ModuleContainerConfig = Field(
        default_factory=lambda: ModuleContainerConfig()
    )
    populations: PopulationsParams = Field(default_factory=lambda: PopulationsParams())
    connections: ConnectionsParams = Field(default_factory=lambda: ConnectionsParams())

    def save_to_json(self, filepath: Path, indent: int = 2) -> None:
        """Serializes the MasterConfig instance to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=indent))
