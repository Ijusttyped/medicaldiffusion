#!/bin/bash -l
#SBATCH --job-name=medDiffusion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=150gb
#SBATCH --gpus=1
#SBATCH --time=48:00:00 # 24 hours
#SBATCH --output=logs/medDiff_%j.log # %j is job id

date
pwd
hostname -f
export PYTHONPATH=$PWD:$PWD
. /dhc/home/marcel.fernandezrosa/conda3/etc/profile.d
python train/train_vqgan.py \
dataset=adni \
model=vq_gan_3d \
run_name=reproduction-adni-vqgan