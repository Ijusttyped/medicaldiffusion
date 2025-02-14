""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse
import glob
import torchio as tio
import pandas as pd


class ADNIDataset(Dataset):
    def __init__(self, root_dir="../ADNI", augmentation=False):
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(root_dir, "adni_annotation.csv"))
        self.sample_size = 998
        self.file_names = self._select_filenames()
        self.augmentation = augmentation
        self.name = "adni"

    def __len__(self):
        return len(self.file_names)

    def roi_crop(self, image):
        # Mask of non-black pixels (assuming image has a single channel).
        mask = image > 0

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)

        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

        # Get the contents of the bounding box.
        cropped = image[x0:x1, y0:y1, z0:z1]

        padded_crop = tio.CropOrPad(np.max(cropped.shape))(cropped.copy()[None])

        padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))

        return padded_crop

    def __getitem__(self, index):
        path = self.file_names[index]
        img = nib.load(path)

        img = np.swapaxes(img.get_data(), 1, 2)
        if len(img.shape) == 4:
            print("Found 4D image: {}".format(path))
        img = np.flip(img, 1)
        img = np.flip(img, 2)
        img = self.roi_crop(image=img)
        sp_size = 64
        img = resize(img, (sp_size, sp_size, sp_size), mode="constant")
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3 * torch.rand(1)[0] + 0.7
            if random_n[0] > 0.5:
                img = np.flip(img, 0)

            img = img * random_i.data.cpu().numpy()

        imageout = torch.from_numpy(img).float().view(1, sp_size, sp_size, sp_size)
        imageout = imageout * 2 - 1

        return {"data": imageout}

    def _select_filenames(self) -> np.ndarray:
        cdr0 = self.data[self.data["CDGLOBAL"] == 0.0]
        self.sample_data = cdr0.sample(self.sample_size, random_state=42)
        return self.sample_data["filepath_MNI"].values
