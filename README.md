This repository contains a simple controller made of only the planner and the feedforward part of the motor cortex.

Models of the neurons making up planner and feedforward motor cortex are generated through NESTML. The module can be compiled by running:
`./nestml/generate_controller_module.py`

The file `./complete_control/complete.music` can be modified to allocate the desired number of slots to both the controller script and the plant one.
`brain.py` constructs the controller and handles its simulation in NEST.
`receiver_plant.py` constructs the plant model and handles its simulation in Bullet.
Helper classes are defined in additional Python scripts.
