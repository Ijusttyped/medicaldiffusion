import argparse
import json
import pytorch_ssim
from torch.utils.data import DataLoader
from train.get_dataset import get_dataset
from hydra import initialize, initialize_config_module, initialize_config_dir, compose


def load_config(config_path="../config/"):
    with initialize(config_path=config_path):
        cfg = compose(
            config_name="base_cfg.yaml",
            overrides=[
                "model=ddpm",
                "dataset=adni",
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


def calculate_ssim(train_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8
    )
    sum_msssim = 0
    num_pairs = len(train_dataset) * (len(train_dataset) - 1) // 2

    for i, sample1 in enumerate(train_loader):
        img1 = sample1["data"][0].cpu()

        for j, sample2 in enumerate(train_loader):
            if i == j:
                continue

            img2 = sample2["data"][0].cpu()
            msssim = pytorch_ssim.msssim_3d(img1, img2)
            sum_msssim += msssim

    average_msssim = sum_msssim / num_pairs
    return average_msssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default="dataset_ssim.json")
    parser.add_argument("--config_path", type=str, default="../config/")

    args = parser.parse_args()
    cfg = load_config(args.config_path)
    train_dataset = load_data(cfg)
    ssim = calculate_ssim(train_dataset)
    print("-------------- SSIM --------------")
    print(ssim)
    with open(args.out_path, "w") as f:
        json.dump({"ssim": ssim.item()}, f)
