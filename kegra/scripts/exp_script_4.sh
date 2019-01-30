#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

# source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate tensorflow_mpi


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=68


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

watchfile=/homes/cx2/gcn/keras-gcn/results/exp_script_4.log

cd /homes/cx2/gcn/keras-gcn/kegra/
# python train_rgcn.py 2>&1 | tee $watchfile
python train_rgcn.py --lr 0.01 --reigen 0 --shuffle 0 --ntrials 40 2>&1 | tee $watchfile
