import os
from dataclasses import dataclass
from pathlib import Path

COMPLETE_CONTROL = Path(__file__).parent.parent.resolve()
ROOT = COMPLETE_CONTROL.parent
RUNS_DIR = ROOT / "runs"  # Base directory for all runs

REFERENCE_DATA_DIR = COMPLETE_CONTROL / "reference_data"

CONFIG = COMPLETE_CONTROL / "config"

TRAJECTORY = CONFIG / "trajectory.txt"
MOTOR_COMMANDS = CONFIG / "motor_commands.txt"
NESTML_BUILD_DIR = ROOT / "nestml" / "target"
CEREBELLUM = ROOT / "cerebellum"
CEREBELLUM_CONFIGS = ROOT / "cerebellum_configurations"
FORWARD = CEREBELLUM_CONFIGS / "forward.yaml"
INVERSE = CEREBELLUM_CONFIGS / "inverse.yaml"
BASE = CEREBELLUM_CONFIGS / "microzones_complete_nest.yaml"
PATH_HDF5 = os.environ.get("BSB_NETWORK_FILE")


@dataclass(frozen=True)
class RunPaths:
    """Holds the standard paths for a single simulation run."""

    run: Path
    data_nest: Path
    data_bullet: Path
    figures: Path
    figures_receiver: Path
    logs: Path


def setup_run_paths(run_timestamp: str):
    """
    Sets up the directory structure for a single simulation run.

    Args:
        run_timestamp: A string timestamp (e.g., YYYYMMDD_HHMMSS).

    Returns:
        RunPaths: A dataclass instance containing Path objects for
                            'run', 'data', 'figures', 'logs'.
    """
    run_dir = RUNS_DIR / run_timestamp
    data_dir = run_dir / "data"
    data_nest_dir = data_dir / "nest"
    data_bullet_dir = data_dir / "bullet"
    figures_dir = run_dir / "figures_pop"
    figures_receiver_dir = run_dir / "figures_rec"
    logs_dir = run_dir / "logs"

    # Create directories if they don't exist
    for dir_path in [
        run_dir,
        data_nest_dir,
        data_bullet_dir,
        figures_dir,
        figures_receiver_dir,
        logs_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    paths_obj = RunPaths(
        run=run_dir,
        data_nest=data_nest_dir,
        data_bullet=data_bullet_dir,
        figures=figures_dir,
        figures_receiver=figures_receiver_dir,
        logs=logs_dir,
    )
    return paths_obj


RUNS_DIR.mkdir(parents=True, exist_ok=True)
