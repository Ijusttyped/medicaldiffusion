import argparse
import json
import torch
from ddpm import Unet3D, GaussianDiffusion, Trainer
import pytorch_ssim
from train.get_dataset import get_dataset
from hydra import initialize, initialize_config_module, initialize_config_dir, compose


def load_config(vqgan_checkpoint, config_path="../config/"):
    with initialize(config_path=config_path):
        cfg = compose(
            config_name="base_cfg.yaml",
            overrides=[
                "model=ddpm",
                "dataset=adni",
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


def calculate_ssim(trainer, num_samples=1000):
    sum_ssim = 0
    for i in range(num_samples):
        trainer.ema_model.eval()
        with torch.no_grad():
            sample = trainer.ema_model.sample(batch_size=2)
        img1 = sample[0].cpu()
        img2 = sample[1].cpu()

        msssim = pytorch_ssim.msssim_3d(img1, img2)
        sum_ssim = sum_ssim + msssim
        print(sum_ssim / (i + 1.0))
    return sum_ssim / num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_checkpoint", type=str)
    parser.add_argument("--vqgan_checkpoint", type=str)
    parser.add_argument("--out_path", type=str, default="ddpm_ssim.json")
    parser.add_argument("--config_path", type=str, default="../config/")

    args = parser.parse_args()
    cfg = load_config(args.vqgan_checkpoint, args.config_path)
    train_dataset = load_data(cfg)
    trainer = load_ddpm(cfg, train_dataset, args.ddpm_checkpoint)
    ssim = calculate_ssim(trainer)
    print("-------------- SSIM --------------")
    print(ssim)
    with open(args.out_path, "w") as f:
        json.dump({"ssim": ssim.item()}, f)
