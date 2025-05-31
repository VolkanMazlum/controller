This folder contains everything needed to compile, run, and test a preliminary implementation of the sinexp and alpha synaptic plasticity model.

The subfolder `custom_stdp` contains the source code for the needed version of the cerebellum that we will test.
The first one contains four E-GLIF models for Granule cell (`eglif_cond_alpha_multisyn`), Inferior Olivary cell (`eglif_io_nestml`), Purkinje cell (`eglif_pc_nestml`) and MLI (`eglif_mli`)

To compile these extension modules, create a build directory at the same level of the folder containing the source code. Inside the `plasticity` folder:
```
mkdir build_dir
```
Go inside it:
```
cd build_dir
```
If it's not the first time you compile the module in this build directory, remove all files:
```
rm -rf *
```
Then compile the module by running:
```
cmake -Dwith-nest={path of your nest-config} {path to custom_stdp folder}
make
make install
```
You should then be able to load the module (For our custom_stdp, module_name should be custom_stdp_module):
```
python3
import nest
nest.Install("module_name")
```

In the folder `tests_plasticity`, there is a battery of tests that can be run with:
```
source do_tests_sinexp.sh
source do_tests_alpha.sh
```
 `do_tests_sinexp.sh` has all the tests for the sinexp synaptic plasticity mode. 
- Base connections: IO -> PC  and GR -> PC
`do_tests_alpha.sh` has all the alpha synaptic plasticity mode tests.
- Base connections: IO -> MLI  and GR -> MLI
