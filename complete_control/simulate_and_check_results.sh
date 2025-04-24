#!/bin/bash
set -euo pipefail # Exit on error, unset var, pipe failure

# this file runs the simulation and ensures the correctness of the result
# it runs the simulation, compares it to a reference (after sorting them)
# i'm perfectly happy to remove it if a better option appears...
# or at least move it somewhere it will be less in the way.

echo "running simulation..."
mpirun -np 8 music complete.music

echo "checking differences..."
python verify_results_file.py ./data/nest/state_p.gdf ./reference_data/reference_state_p.gdf
python verify_results_file.py ./data/nest/state_n.gdf ./reference_data/reference_state_n.gdf
