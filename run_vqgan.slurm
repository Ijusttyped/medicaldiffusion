#!/bin/bash -l
#SBATCH --job-name=medDiffusion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=3 # -c
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
dataset.root_dir='/dhc/groups/fglippert/adni_t1_mprage' \
model=vq_gan_3d \
model.gpus=1 \
model.batch_size=2 \
model.n_codes=16384 \
run_name=test-adni-vqgan-01