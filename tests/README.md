
---

# SinExp and Alpha Synaptic Plasticity Model

This repository contains everything required to **compile, run, and test** a preliminary implementation of the **SinExp** and **Alpha synaptic plasticity models**.

---

## Directory Structure

* The subfolder custom_stdp (in the repo) contains the source code for the needed version of the cerebellum that we will test. The first one contains four E-GLIF models for Granule cell (eglif_cond_alpha_multisyn), Inferior Olivary cell (eglif_io_nestml), Purkinje cell (eglif_pc_nestml) and MLI (eglif_mli)

* Subfolders include E-GLIF models:

  * `eglif_cond_alpha_multisyn`: Granule Cell
  * `eglif_io_nestml`: Inferior Olivary Cell
  * `eglif_pc_nestml`: Purkinje Cell
  * `eglif_mli_nestml`: Molecular Layer Interneuron (MLI) and more ...

---

## Compilation Instructions

To compile the extension modules:

1. **Create a build directory** (at the same level as the source folder (inside the tests folder)):

   ```
   mkdir build_dir
   cd build_dir
   ```

2. **Clean previous builds** (only if rebuilding):

   ```
   rm -rf *
   ```

3. **Run CMake with appropriate paths**:

   ```
   cmake -Dwith-nest=<path_to_your_nest> <path_to_custom_stdp>
   make
   make install
   ```

---

## Using the Module in Python

To load the compiled module:

```python
import nest
nest.Install("custom_stdp_module")  # Replace with actual module name if different
```

---

## Running Tests

To test neuron connections and plasticity logic:

1. **Install the required Python library**:

   ```
   pip install pynest
   ```

2. **Navigate to the test directory** (go to the same directory with e.g., `tests_plasticity/` or `test_neurons/`).

3. **Run tests using `pytest`**:

   ```
   pytest
   ```

