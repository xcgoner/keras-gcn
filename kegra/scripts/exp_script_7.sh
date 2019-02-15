#!/bin/bash
#PBS -l select=1:ncpus=112 -lplace=excl

# source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate tensorflow_mpi


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=56


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

# basename=/homes/cx2/gcn/keras-gcn/results/exp_script_4

# watchfile=$basename.log
# modelfile=$basename.h5

# cd /homes/cx2/gcn/keras-gcn/kegra/
# # python train_rgcn.py 2>&1 | tee $watchfile
# python train_gcn_dg.py --save $modelfile --lr 0.01 --nfolds 0 --augmentation no_augmentation --nepochs 500 --nlayers 2 --selfloop eye --ntrials 10 2>&1 | tee $watchfile


basedir=/homes/cx2/gcn/keras-gcn/results
basename=gcn_exp
dataset="citeseer"
# percent=0.03
lr=0.03
nlayers=2

watchfile1=${basedir}/${basename}_7.log

for nfilters in 16 64
do
    for percent in 0.01 0.007 0.005 0.003
    do
        for expm in 1 2
        do
            watchfile=${basedir}/${basename}_${dataset}_${percent}_${lr}_${nlayers}_${nfilters}_${expm}.log
            modelfile=${basedir}/model.${basename}_${dataset}_${percent}_${lr}_${nlayers}_${nfilters}_${expm}.h5
            cd /homes/cx2/gcn/keras-gcn/kegra/
            python train_gcn_exponential_1.py --save ${modelfile} --dataset ${dataset} --train-percent ${percent} --append 0 --lr ${lr} --nepochs 200 --nlayers ${nlayers} --nfilters ${nfilters} --expm ${expm} --ntrials 10 2>&1 | tee ${watchfile} ${watchfile1}
        done
    done
done