# NRP-Core Experiment Setup and Customization

## Installation Instructions

### 1. Install NRP-Core

Follow the instructions provided here: [NRP-Core Installation Guide](https://neurorobotics.net/Documentation/latest/nrp-core/page_installation.html).

The experiment is designed to work with **nest-server v3.7**. Ensure that you have **nest-simulator** installed beforehand, as NRP-Core natively installs only **nest-simulator v3.1**. When installing NRP-Core, provide the path where nest-simulator is installed to CMake using the additional parameter `-DNEST_INSTALL_DIR=<nest-simulator/install/path>`. 

For example, the corresponding line from the installation section will look like:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX="${NRP_INSTALL_DIR}" -DNRP_DEP_CMAKE_INSTALL_PREFIX="${NRP_DEPS_INSTALL_DIR}" -DNEST_INSTALL_DIR=<nest-simulator/install/path>
```

Skip the installation of some dependencies like Gazebo, Edlut, ROS, MQTT, and NEST (since it is preinstalled). The updated CMake line will look like:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX="${NRP_INSTALL_DIR}" -DNRP_DEP_CMAKE_INSTALL_PREFIX="${NRP_DEPS_INSTALL_DIR}" -DNEST_INSTALL_DIR=<nest-simulator/install/path> -DENABLE_MQTT=OFF -DENABLE_ROS=OFF -DENABLE_GAZEBO=OFF -DENABLE_EDLUT=OFF -DENABLE_SPINNAKER=OFF
```

To speed up the build, you can also add `-DENABLE_TESTING=OFF -DBUILD_RST=OFF`:

```sh
cmake .. -DCMAKE_INSTALL_PREFIX="${NRP_INSTALL_DIR}" -DNRP_DEP_CMAKE_INSTALL_PREFIX="${NRP_DEPS_INSTALL_DIR}" -DNEST_INSTALL_DIR=<nest-simulator/install/path> -DENABLE_MQTT=OFF -DENABLE_ROS=OFF -DENABLE_GAZEBO=OFF -DENABLE_EDLUT=OFF -DENABLE_SPINNAKER=OFF -DENABLE_TESTING=OFF -DBUILD_RST=OFF
```

### 2. Running the Example Experiment

If the installation was successful and NRP-Core is properly installed, you should be able to run the example experiment:

```sh
NRPCoreSim -c try_bullet/simulation_config_nest.json
```

### 3. Useful Documentation

To start working with NRP-Core, it is useful to read the documentation at the following links:
- [NRP-Core Architecture Overview](https://neurorobotics.net/Documentation/latest/nrp-core/page_architecture_overview.html)
- [NRP-Core Main Elements](https://neurorobotics.net/Documentation/latest/nrp-core/page_main_elements.html)

## Modifying the Experiment

### Configuration Files

- **try_bullet/simulation_config_nest.json**:
  Contains the settings for the experiment. You can read more about the file here: [Simulation Configuration](https://neurorobotics.net/Documentation/latest/nrp-core/page_simulation_configuration.html)

### Engine Interfaces

- **try_bullet/bullet_interface.py**:
  This is an instance of the engine (PySim) that interfaces with pyBullet. You can read about the structure of this file here: [PySim Engine](https://neurorobotics.net/Documentation/latest/nrp-core/page_pysim_engine.html)

  Here we define the datapacks ([Datapacks](https://neurorobotics.net/Documentation/latest/nrp-core/page_datapacks.html)) and their structure, which are transferred to and from pyBullet. In particular, in `runLoop`, the actions are converted to pyBullet functions. More details on what is happening can be found in `${NRP_INSTALL_DIR}/lib/python3.8/site-packages/nrp_core/engines/py_sim/BulletLib.py` in the `run_one_step` function.

### NEST Engine

- **try_bullet/nest_engine.py**:
  This is the initialization file for the NEST server. Everything before `nest.Simulate()` happens here. The `Simulate()` step is handled by NRP-Core. The outputs from the NEST server, as an example, are defined at the end of this file.

### Data Transformation

- **try_bullet/tf_from_nest_to_bullet.py**, **try_bullet/tf_from_bullet.py**:
  These files process the outputs from/to NEST/pyBullet engines and pass them forward. For example, the spike rate is calculated and passed as an action to pyBullet. How the actions data is unpacked is defined in `try_bullet/bullet_interface.py` in `runLoop()`.

### Additional Components

- **try_bullet/controller_module.so**:
  This should be compiled with nestml.

- **trajectory.txt** and **motor_commands.txt**:
  These files are to be generated. The length of the generated parameters should correspond to the length of the neurons' history in `nest_engine.py`. Probably this initialization could be moved to `nest_engine.py`.

## Conclusion

By following the above steps, you should be able to successfully install NRP-Core and run the provided example experiment. The documentation links and configuration details will assist you in modifying and customizing the experiment to suit your needs. If you encounter any issues, refer to the official [NRP-Core documentation](https://neurorobotics.net/Documentation/latest/nrp-core/index.html) for further guidance.
