#!/bin/bash -l
#SBATCH --job-name=medDiffusion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --partition=gpua100,gpupro,gpu # -p
#SBATCH --cpus-per-task=3 # -c
#SBATCH --mem=150gb
#SBATCH --gpus=1
#SBATCH --time=72:00:00 # 24 hours
#SBATCH --output=logs/medDiff_%j.log # %j is job id

date
pwd
hostname -f
export PYTHONPATH=$PWD:$PWD
. /dhc/home/marcel.fernandezrosa/conda3/etc/profile.d
python train/train_ddpm.py \
model=ddpm \
dataset=adni \
model.results_folder_postfix='flair' \
model.vqgan_ckpt='/dhc/home/marcel.fernandezrosa/medicaldiffusion/output/checkpoints/vq_gan/ADNI/flair/lightning_logs/version_589904/checkpoints/latest_checkpoint.ckpt' \
dataset.root_dir='/dhc/groups/fglippert/adni_t1_mprage' \
model.diffusion_img_size=32 \
model.diffusion_depth_size=32 \
model.diffusion_num_channels=8 \
model.dim_mults=[1,2,4,8] \
model.batch_size=10 \
model.gpus=1