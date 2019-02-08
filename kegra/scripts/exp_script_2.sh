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

# basename=/homes/cx2/gcn/keras-gcn/results/exp_script_2

# watchfile=$basename.log
# modelfile=$basename.h5

# cd /homes/cx2/gcn/keras-gcn/kegra/
# python train_gcn_exponential_1.py --save $modelfile --dataset "cora" --train-percent 0.01 --lr 0.04 --nepochs 200 --nlayers 3 --nfilters 16 --expm 2 --ntrials 10 2>&1 | tee $watchfile


basename=/homes/cx2/gcn/keras-gcn/results/gcn_exp
dataset="cora"
percent=0.02
lr=0.02

watchfile1=${basename}_2.log

for nlayers in 1 2 3
do
    for expm in 1 2 3
    do
        watchfile=${basename}_${dataset}_${percent}_${lr}_${nlayers}_${expm}.log
        modelfile=${basename}_${dataset}_${percent}_${lr}_${nlayers}_${expm}.h5
        cd /homes/cx2/gcn/keras-gcn/kegra/
        python train_gcn_exponential_1.py --save ${modelfile} --dataset ${dataset} --train-percent ${percent} --lr ${lr} --nepochs 200 --nlayers ${nlayers} --nfilters 16 --expm ${expm} --ntrials 10 2>&1 | tee ${watchfile} ${watchfile1}
    done
done