import argparse
import torch
import nibabel as nib
import numpy as np
from ddpm import Unet3D, GaussianDiffusion, Trainer
from train.get_dataset import get_dataset
from hydra import initialize, initialize_config_module, initialize_config_dir, compose


def load_config(vqgan_checkpoint, dataset_dir, config_path="../config/"):
    with initialize(config_path=config_path):
        cfg = compose(
            config_name="base_cfg.yaml",
            overrides=[
                "model=ddpm",
                "dataset=adni",
                f"dataset.root_dir={dataset_dir}",
                f"model.vqgan_ckpt={vqgan_checkpoint}",
                "model.diffusion_img_size=32",
                "model.diffusion_depth_size=32",
                "model.diffusion_num_channels=8",
                "model.dim_mults=[1,2,4,8]",
                "model.batch_size=40 ",
                "model.gpus=1",
            ],
        )
    return cfg


def load_data(cfg):
    train_dataset, _, _ = get_dataset(cfg)
    return train_dataset


def load_ddpm(cfg, train_dataset, ddpm_checkpoint):
    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        # logger=cfg.model.logger
    )
    trainer.load(ddpm_checkpoint, map_location="cuda:0")
    return trainer


def generate(trainer):
    trainer.ema_model.eval()
    with torch.no_grad():
        sample = trainer.ema_model.sample(batch_size=1)
    return sample[0].squeeze(dim=0).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_checkpoint", type=str)
    parser.add_argument("--vqgan_checkpoint", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--out_path", type=str, default="sample.nii.gz")
    parser.add_argument("--config_path", type=str, default="../config/")

    args = parser.parse_args()
    cfg = load_config(args.vqgan_checkpoint, args.dataset_dir, args.config_path)
    train_dataset = load_data(cfg)
    trainer = load_ddpm(cfg, train_dataset, args.ddpm_checkpoint)
    seq = generate(trainer)
    img = nib.Nifti1Image(seq, affine=np.eye(4))
    nib.save(img, args.out_path)
    print(f"Stored sample in {args.out_path}")
