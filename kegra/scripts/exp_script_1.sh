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

# basename=/homes/cx2/gcn/keras-gcn/results/exp_script_1

# watchfile=$basename.log
# modelfile=$basename.h5

# cd /homes/cx2/gcn/keras-gcn/kegra/
# # python train_rgcn.py 2>&1 | tee $watchfile
# python train_gcn_reg.py --save $modelfile --lr 0.001 --nfolds 0 --augmentation no_augmentation --nfilters 64 --selfloop "eye" --reigen 0.1 --nepochs 400 --nlayers 2 --ntrials 1 2>&1 | tee $watchfile
# # python train_gcn_reg.py --save $modelfile --lr 0.002 --nfolds 0 --augmentation no_augmentation --nfilters 64 --selfloop "eye" --reigen 0 --nepochs 200 --nlayers 2 --ntrials 1 2>&1 | tee $watchfile

basename=/homes/cx2/gcn/keras-gcn/results/gcn_exp
dataset="cora"
percent=0.03
lr=0.01

watchfile1=${basename}_1.log

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
