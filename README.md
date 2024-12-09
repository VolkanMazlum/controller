This branch of the repository features all code needed to run simulations of reaching tasks on a virtual robotic arm driven by a closed-loop cerebellar controller.
Models of the neurons making up planner, feedforward motor cortex and brainstem are generated through NESTML. The module can be compiled by running:
`./nestml/generate_controller_module.py`

`brain.py` constructs the controller and handles its simulation in NEST.
`receiver_plant.py` constructs the plant model and handles its simulation in Bullet.
Helper classes are defined in additional Python scripts.

The file `./complete_control/complete.music` can be modified to allocate the desired number of slots to both the controller script and the plant one. The simulation can be started by running:
`mpirun -np 2 music complete.music` 
inside ./complete_control. The value of the -np parameter should be adjusted according to the number of processes allocated in the `complete.music` file.

