#!/bin/bash -l
#SBATCH --job-name=medDiffusion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=20 # -c
#SBATCH --mem=150gb
#SBATCH --gpus=1
#SBATCH --time=48:00:00 # 24 hours
#SBATCH --output=logs/medDiff_%j.log # %j is job id

date
pwd
hostname -f
export PYTHONPATH=$PWD:$PWD
. /dhc/home/marcel.fernandezrosa/conda3/etc/profile.d
python train/train_ddpm.py \
dataset=adni \
model=ddpm \
model.vqgan_ckpt='/dhc/home/marcel.fernandezrosa/medicaldiffusion/output/checkpoints/vq_gan/ADNI/T1toMNI/reproduction-adni-vqgan_MedicalDiffusion/4_251p1hyp/checkpoints/latest_checkpoint.ckpt' \
model.diffusion_img_size=32 \
model.diffusion_depth_size=32 \
model.diffusion_num_channels=8 \
model.dim_mults=[1,2,4,8] \
run_name=reproduction-adni-ddpm