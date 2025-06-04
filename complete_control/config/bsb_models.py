from pathlib import Path

from pydantic import BaseModel, FilePath

from . import paths


class BSBConfigPaths(BaseModel, frozen=True):
    forward_yaml: FilePath = paths.FORWARD
    inverse_yaml: FilePath = paths.INVERSE
    base_yaml: FilePath = paths.BASE
    cerebellum_hdf5: FilePath = paths.PATH_HDF5


class BSBConfigCopies(BaseModel, frozen=True):
    forward_yaml_content: str = ""
    inverse_yaml_content: str = ""
    microzones_complete_nest_yaml_content: str = ""
