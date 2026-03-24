import torch
from monai.data import DataLoader, Dataset
import zarr
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld,
    ToTensord,
)


class Zarr2DDataset(Dataset):
    def __init__(self, zarr_path):
        print(f"Loading dataset from: {zarr_path}")
        self.zarr_file = zarr.open(zarr_path, mode="r")
        self.keys = list(self.zarr_file.keys())
        self.index_map = []

        for key in self.keys:
            grp = self.zarr_file[key]

            raw = grp["raw"]  # (S, H, W)

            if "mask" in grp:
                mask = grp["mask"]  # (S, H, W)
            else:
                mask = np.zeros_like(raw, dtype=np.uint8)

            z_size = raw.shape[0]
            for z in range(z_size):
                self.index_map.append((key, z))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        key, z = self.index_map[idx]
        grp = self.zarr_file[key]

        raw2d = np.asarray(grp["raw"][z], dtype=np.float32)  # (H, W)

        if "mask" in grp:
            mask2d = np.asarray(grp["mask"][z], dtype=np.float32)  # (H, W)
        else:
            mask2d = np.zeros_like(raw2d, dtype=np.float32)

        mask2d = (mask2d > 0).astype(np.float32)

        raw2d = np.expand_dims(raw2d, axis=0)    # (1, H, W)
        mask2d = np.expand_dims(mask2d, axis=0)  # (1, H, W)

        return {"image": raw2d, "label": mask2d, "key": key, "slice_idx": z}


def build_unet(in_channels=1, out_channels=1):
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_small_unet(in_channels=1, out_channels=1):
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(16, 32, 64),
        strides=(1, 1),
        num_res_units=0,
    )


class zarr3Ddataset(Dataset):
    def __init__(self, zarr_path):
        self.zarr_file = zarr.open(zarr_path, mode='r')
        self.keys = list(self.zarr_file.keys())
        self.index_map = []
        for k in self.keys:
            raw = np.asarray(self.zarr_file[k]["raw"])
            for i in range(raw.shape[0]):
                self.index_map.append((k, i))

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        key, i = self.index_map[idx]
        grp = self.zarr_file[key]

        mask2d = grp["mask"][i] # (H, W)
        raw2d = grp["raw"][i]   # (H, W)

        label = np.expand_dims(mask2d, axis=0)
        raw2d = np.expand_dims(raw2d, axis=0).astype(np.float32)

        return {"image": raw2d, "label":label, "key": key, "slice_idx": i}

PH, PW = 192, 192
trfm = Compose([
    SpatialPadd(keys=["image", "label"], spatial_size=(PH, PW), mode="constant"),
    ScaleIntensityRangePercentilesd(
        keys="image",
        lower=1,
        upper=99,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    # RandCropByPosNegLabeld(
    #     keys=["image", "label"],
    #     label_key="label",
    #     spatial_size=(128, 128),
    #     pos=0.7, neg=0.3,
    #     num_samples=4,
    # ),
    ToTensord(keys=["image", "label"]),
])


@torch.no_grad()
def validate_model(
    checkpoint_filename,
    dataset_filename,
    save_filename,
    device="cpu",
    do_plot=True,
    save_probs=True,
    plot_slice=None,
    do_student_unet=True,
):
    # Load checkpoint
    chk = torch.load(checkpoint_filename, map_location=device, weights_only=True)

    # Build network and Load dataset
    if do_student_unet:
        net = build_unet().to(device)
        net.load_state_dict(chk["student_state_dict"], strict=True)
        ds = zarr3Ddataset(dataset_filename)
    else:
        net = build_small_unet().to(device)
        net.load_state_dict(chk["network_state_dict"], strict=True)
        ds = Zarr2DDataset(dataset_filename)

    net.eval()

    test_ds = Dataset(ds, transform=trfm)
    loader = DataLoader(test_ds, shuffle=False, batch_size=1)

    zarr_f = zarr.open(save_filename, mode="a")

    for i, batch in enumerate(loader):
        x = batch["image"].to(device).float()
        l = batch["label"].to(device).float()
        logits = net(x)
        preds = torch.sigmoid(logits) if save_probs else logits

        x_np = x.detach().cpu().numpy()[0][0]      # (H, W)
        y_np = preds.detach().cpu().numpy()[0][0]  # (H, W)
        l_np = l.detach().cpu().numpy()[0][0]  # (H, W)
        pred2d_labels = (y_np > 0.8).astype(np.uint8)

        key = batch["key"][0] if isinstance(batch["key"], list) else batch["key"]
        slice_idx = int(batch["slice_idx"][0]) if "slice_idx" in batch else i

        # Save predictions
        case_group = zarr_f.require_group(str(key))
        slice_group = case_group.require_group(str(slice_idx))
        slice_group.create_array("image", data=x_np, overwrite=True)
        slice_group.create_array("pred_labels", data=pred2d_labels, overwrite=True)
        slice_group.create_array("pred_probs", data=y_np, overwrite=True)

        # Plot predictions
        if do_plot:
            img2d = x_np
            pred2d = (y_np > 0.8).astype(np.uint8)
            # pred2d = y_np

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(img2d, cmap="gray")
            ax[0].imshow(l_np, cmap="jet", alpha=0.5)

            ax[0].set_title(f"Image | case {key} | slice {slice_idx}")
            ax[0].axis("off")

            ax[1].imshow(img2d, cmap="gray")
            ax[1].imshow(pred2d, cmap="jet", alpha=0.5)
            ax[1].set_title("Prediction overlay")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    checkpoint_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\training_2d_notFlattening_splitData\best_model_v1.pth"
    dataset_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\test_data_segmentation\someSample_CytoT2b.zarr"
    save_filename = r"Z:\bmahsa\workplaces\CellFundusSegmentation\test_data_segmentation\segmentation_someSample_CytoT2b.zarr"

    validate_model(
        checkpoint_filename,
        dataset_filename,
        save_filename,
        save_probs=True,
        do_plot=False,
    )