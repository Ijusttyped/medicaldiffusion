import argparse
import json
import torch
from torch.utils.data import DataLoader
from vq_gan_3d.model.vqgan import VQGAN
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


def load_model_from_checkpoint(vqgan_checkpoint):
    vqgan = VQGAN.load_from_checkpoint(vqgan_checkpoint)
    return vqgan


def calculate_ssim(vqgan, train_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqgan = vqgan.to(device)
    sum_ssim = 0
    num_samples = len(train_dataset)
    for i, image in enumerate(train_loader):
        vqgan.eval()
        with torch.no_grad():
            pred_image = vqgan(image["data"].to(device))
        img1 = image["data"][0].cpu()
        img2 = pred_image[1][0].cpu()
        msssim = pytorch_ssim.msssim_3d(img1, img2)
        sum_ssim = sum_ssim + msssim
        print(sum_ssim / (i + 1.0))
    return sum_ssim / num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqgan_checkpoint", type=str)
    parser.add_argument("--out_path", type=str, default="vqgan_ssim.json")
    parser.add_argument("--config_path", type=str, default="../config/")

    args = parser.parse_args()
    cfg = load_config(args.vqgan_checkpoint, args.config_path)
    train_dataset = load_data(cfg)
    vqgan = load_model_from_checkpoint(args.vqgan_checkpoint)
    ssim = calculate_ssim(vqgan, train_dataset)
    print("-------------- SSIM --------------")
    print(ssim)
    with open(args.out_path, "w") as f:
        json.dump({"ssim": ssim.item()}, f)
