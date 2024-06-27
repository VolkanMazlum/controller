This branch of the repository contains a simple controller made of only the planner, the feedforward part of the motor cortex and the brainstem.

Models of the neurons making up the network are generated through NESTML. The module can be compiled by running:
`python3 ./nestml/generate_controller_module.py`
This will create a `target` folder containing the compiled module that can be installed in NEST.

The script `generate_trajectories.py` computes the input signals and writes them into `trajectory.txt` and '`motor_commands.txt` (both already provided in the repo).

The script `simulation.py` sets up the network, runs the simulation in NEST, computes the output firing rate and writes it to file. Inputs and outputs of each block are furtherly listed in this script as well.

