#!/usr/bin/env python3

__authors__ = "Massimo Grillo"
__copyright__ = "Copyright 2021"
__credits__ = ["Massimo Grillo"]
__license__ = "GPL"
__version__ = "1.0.1"

import numpy as np
import matplotlib.pyplot as plt
#import nest
import mpi4py

#from bsb.core import from_hdf5
from bsb.output import HDF5Formatter
from bsb.config import JSONConfig
from bsb.reporting import set_verbosity


# Reconfigure scaffold
filename_h5 = "300x_200z_claudia_dcn_test_3.hdf5"
filename_config = 'mouse_cerebellum_cortex_update_dcn_copy_post_stepwise_colonna_X.json'
if mpi4py.MPI.COMM_WORLD.rank == 0:
    reconfigured_obj = JSONConfig(filename_config)
    HDF5Formatter.reconfigure(filename_h5, reconfigured_obj)
    print('HDF5 reconfigured!!')
