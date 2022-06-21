#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N GG_TEST
#PBS -M gustavo.giardini@iit.it
#PBS -m e
#PBS -q gpu

cd $PBS_O_WORKDIR
unset CUDA_VISIBLE_DEVICES
singularity run --nv -c -B /work/ggiardini/kuka-ml-threading/ kuka-ml.sif $WORKDIR/kuka-ml-threading/train_models_generate_confusion_matrix.py > output_test.txt
