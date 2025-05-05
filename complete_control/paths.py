from pathlib import Path

COMPLETE_CONTROL = Path(__file__).parent.resolve()

ROOT = COMPLETE_CONTROL.parent

FIGURE_DIR = COMPLETE_CONTROL / "fig"
FIGURES_THESIS_DIR = COMPLETE_CONTROL / "figures_thesis"
REFERENCE_DATA_DIR = COMPLETE_CONTROL / "reference_data"
LOG_DIR = COMPLETE_CONTROL / "logs"
DATA_DIR = COMPLETE_CONTROL / "data"
NEST_DATA_DIR = DATA_DIR / "nest"
BULLET_DATA_DIR = DATA_DIR / "bullet"

# ensure paths exist
for dir in [NEST_DATA_DIR, BULLET_DATA_DIR, LOG_DIR, FIGURE_DIR, FIGURES_THESIS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# specific files
TRAJECTORY = COMPLETE_CONTROL / "trajectory.txt"
MOTOR_COMMANDS = COMPLETE_CONTROL / "motor_commands.txt"
