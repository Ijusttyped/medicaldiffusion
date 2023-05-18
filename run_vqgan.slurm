#!/bin/bash -l
#SBATCH --job-name=medDiffusion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --partition=gpua100,gpupro # -p
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
python train/train_vqgan.py \
dataset=adni \
dataset.root_dir='/dhc/groups/fglippert/adni_t1_mprage' \
model=vq_gan_3d \
model.gpus=1 \
model.default_root_dir_postfix='flair' \
model.precision=16 model.embedding_dim=8 \
model.n_hiddens=16 \
model.downsample=[2,2,2] \
model.num_workers=32 \
model.gradient_clip_val=1.0 \
model.lr=3e-4 \
model.discriminator_iter_start=10000 \
model.perceptual_weight=4 \
model.image_gan_weight=1 \
model.video_gan_weight=1 \
model.gan_feat_weight=4 \
#model.batch_size=2 \
model.n_codes=16384 \
#model.accumulate_grad_batches=1