import json
from typing import Any

import numpy as np
import paths
import settings
import structlog
from settings import Brain, Experiment, MusicCfg, Simulation


class PlantConfig:
    """
    Manages all configuration parameters for the robotic plant simulation.
    """

    def __init__(
        self,
        run_paths: paths.RunPaths,
    ):
        self.log = structlog.get_logger(type(self).__name__)  # TODO configure later
        self.log.info("Initializing PlantConfig...")

        self.run_paths: paths.RunPaths = run_paths
        self.params_json_path = paths.PARAMS
        self.trajectory_path = paths.TRAJECTORY

        # Load settings from new_params.json
        try:
            with open(self.params_json_path) as f:
                self.params_from_json: dict[str, Any] = json.load(f)
            self.log.debug(
                "Loaded parameters from JSON", path=str(self.params_json_path)
            )
        except FileNotFoundError:
            self.log.error(
                "Parameters JSON file not found", path=str(self.params_json_path)
            )
            raise
        except json.JSONDecodeError:
            self.log.error(
                "Error decoding parameters JSON file", path=str(self.params_json_path)
            )
            raise

        # Initialize settings objects from complete_control.settings
        self.sim_settings = Simulation()
        self.exp_settings = Experiment()
        self.brain_settings = Brain()
        self.music_settings = MusicCfg()

        # --- Extract and Store Key Parameters ---
        self.SEED: int = settings.SEED
        np.random.seed(self.SEED)  # Seed numpy as early as possible

        # Module-specific params (can be accessed via self.params_from_json['modules']['module_name'])
        self.module_params: dict[str, Any] = self.params_from_json["modules"]
        self.spine_params: dict[str, Any] = self.module_params["spine"]

        # Simulation timing
        self.RESOLUTION_MS: float = self.sim_settings.resolution
        self.RESOLUTION_S: float = self.RESOLUTION_MS / 1000.0
        self.TIME_MAX_MS: float = self.sim_settings.timeMax
        self.TIME_MAX_S: float = self.TIME_MAX_MS / 1000.0
        self.TIME_WAIT_MS: float = self.sim_settings.timeWait
        self.TIME_WAIT_S: float = self.TIME_WAIT_MS / 1000.0
        self.N_TRIALS: int = self.sim_settings.n_trials

        self.TIME_TRIAL_S: float = self.TIME_MAX_S + self.TIME_WAIT_S
        self.TOTAL_SIM_DURATION_S: float = self.TIME_TRIAL_S * self.N_TRIALS
        self.time_vector_total_s: np.ndarray = np.arange(
            0, self.TOTAL_SIM_DURATION_S, self.RESOLUTION_S
        )
        # Single trial time vector (for one segment of activity + wait)
        self.time_vector_single_trial_s: np.ndarray = np.arange(
            0, self.TIME_TRIAL_S, self.RESOLUTION_S
        )

        # Experiment and Robot
        self.NJT: int = self.exp_settings.dynSys.numVariables()
        self.DYN_SYS = self.exp_settings.dynSys
        self.CONNECT_GUI = False

        self.initial_joint_pos_rad: float = self.exp_settings.init_pos_angle
        self.target_joint_pos_rad: float = self.exp_settings.tgt_pos_angle

        self.N_NEURONS: int = self.brain_settings.nNeurPop

        # Plant interaction parameters
        self.SCALE_TORQUE: float = 500000.0  # From original receiver_plant.py
        self.BUFFER_SIZE_S: float = (
            10.0 / 1000.0
        )  # Buffer to calculate spike rate (seconds)

        # MUSIC configuration
        self.MUSIC_CONST_S: float = self.music_settings.const / 1000.0
        self.MUSIC_ACCEPTABLE_LATENCY_S: float = 2 * self.RESOLUTION_S - (
            self.RESOLUTION_S - self.MUSIC_CONST_S
        )
        if self.MUSIC_ACCEPTABLE_LATENCY_S < 0:
            self.MUSIC_ACCEPTABLE_LATENCY_S = 0.0
        self.MUSIC_PORT_MOT_CMD_IN: str = (
            "mot_cmd_in"  # Corresponds to "out_port" in main_simulation's music_cfg
        )
        self.MUSIC_PORT_FBK_OUT: str = (
            "fbk_out"  # Corresponds to "in_port" in main_simulation's music_cfg
        )

        # Sensory Neuron parameters from spine_params
        self.SENS_NEURON_BASE_RATE: float = self.spine_params.get(
            "sensNeur_base_rate",
            # 0.0,
        )
        self.SENS_NEURON_KP: float = self.spine_params.get(
            "sensNeur_kp",
            # 1200.0,
        )
        self.SENS_NEURON_ID_START: int = self.brain_settings.firstIdSensNeurons

        # Weight for motor command calculation (motor cortex - motor neurons)
        self.WGT_MOTCTX_MOTNEUR: float = self.spine_params.get(
            "wgt_motCtx_motNeur",
            # 1.0,
        )

        self.log.info("PlantConfig initialized successfully")
        self.log.debug("Config details", config_vars=self.__dict__)
