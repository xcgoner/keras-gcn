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

basename=/homes/cx2/gcn/keras-gcn/results/exp_script_4

watchfile=$basename.log
modelfile=$basename.h5

cd /homes/cx2/gcn/keras-gcn/kegra/
# python train_rgcn.py 2>&1 | tee $watchfile
python train_gcn_dg.py --save $modelfile --lr 0.01 --nfolds 0 --augmentation no_augmentation --nepochs 200 --ntrials 10 2>&1 | tee $watchfile