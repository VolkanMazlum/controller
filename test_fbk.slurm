#! /bin/bash -l
#SBATCH --job-name=massimo
#SBATCH --account=ich027
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=benedetta.gambosi@polimi.it
#SBATCH --constraint=mc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load daint-mc
module switch PrgEnv-cray/6.0.9 PrgEnv-gnu/6.0.9
module load GSL/2.5-CrayGNU-20.11
module load cray-python/3.8.2.1
module load cray-hdf5/1.12.0.0
module load CMake/3.14.5
export PATH=$HOME/bin/bin:$HOME/.local/bin:$PATH
export CRAYPE_LINK_TYPE=dynamic
source /users/bp002003/workspace/nest-simulator-2.18.0-build-mpi/bin/nest_vars.sh

export PATH=/users/bp002003/workspace/MUSIC-build/bin/:$PATH
export PATH=/users/bp002003/workspace/Codici/control_loops/:$PATH
export PYTHONPATH=/users/bp002003/workspace/MUSIC-build/lib/python3.8/site-packages:$PYTHONPATH

cd $SLURM_SUBMIT_DIR

echo "Launching job $SLURM_JOB_NAME with $SLURM_JOB_NUM_NODES nodes and a total of $SLURM_CPUS_PER_TASK threads!"

srun -c${SLURM_CPUS_PER_TASK} music ./fbk_control/fbk.music
