import argparse
import json
import random
from pathlib import Path

import torch
import dnnlib
from evaluation.evaluate_ddpm_ssim import load_config, load_data, load_ddpm
from metrics.frechet_inception_distance import compute_fid
from metrics.metric_utils import MetricOptions


def calculate_fid(trainer, num_samples=2000):
    dataset_kwargs = {
        "class_name": "dataset.adni.ADNIDataset",
        "root_dir": "/dhc/groups/fglippert/adni_t1_mprage",
        "augmentation": False,
    }

    # print('Loading networks from "%s"...' % gan_checkpoint)
    # device = torch.device("cuda")
    # with dnnlib.util.open_url(gan_checkpoint) as f:
    #     G = legacy.load_network_pkl(f)["G_ema"].to(device)

    print(f"Calculating FID for {num_samples} generated images...")
    opts = MetricOptions(G=None, G_kwargs={}, dataset_kwargs=dataset_kwargs)
    fid_val = compute_fid(
        opts=opts,
        trainer=trainer,
        max_real=None,
        num_gen=num_samples,
    )
    return fid_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_checkpoint", type=str)
    parser.add_argument("--vqgan_checkpoint", type=str)
    parser.add_argument("--out_path", type=str, default="ddpm_fid.json")
    parser.add_argument("--config_path", type=str, default="../config/")

    args = parser.parse_args()
    cfg = load_config(args.vqgan_checkpoint, args.config_path)
    train_dataset = load_data(cfg)
    ddpm_trainer = load_ddpm(cfg, train_dataset, args.ddpm_checkpoint)
    fid = calculate_fid(ddpm_trainer)
    print("-------------- FID --------------")
    print(fid)
    with open(args.out_path, "w") as f:
        json.dump({"FID": fid}, f)
