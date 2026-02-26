import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class zarr3Ddataset(Dataset):
    def __int__(self, zarr_path, depth_size):
        self.zarr_file = zarr.open(zarr_path, mode='r')
        self.keys = list(self.zarr_file.keys())
        self.depth_size = depth_size

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        grp = self.zarr_file[key]

        mask3d = grp["mask"][1] # (S, H, W) Frame 1 has information
        raw = grp["raw"][1]   # (S, H, W)

        mask2d = mask3d.max(axis=0)


