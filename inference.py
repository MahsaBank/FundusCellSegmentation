import torch
from monai.data import DataLoader, Dataset
import zarr
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, DivisiblePadd, ScaleIntensityRangePercentilesd, ToTensord
)


def build_unet(in_channels=1, out_channels=1):
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

def make_label_valid(mask3d, safe_dis=12, radius=5):
    mask3d = np.asarray(mask3d)
    mask3d = (mask3d > 0).astype(np.uint8)
    roi = mask3d.astype(bool)
    label = roi.astype(np.float32)
    valid = roi.astype(np.float32)

    # Z, H, W = mask3d.shape
    # valid = np.zeros((Z, H, W), dtype=np.float32)
    # labeled_slices = np.where((mask3d.sum(axis=(1, 2))) > 0)[0]

    # if radius > 0 and len(labeled_slices) > 0:
    #     slice_w = np.zeros((Z,), dtype=np.float32)
    #     for z0 in labeled_slices:
    #         for dz in range(-radius, radius + 1):
    #             z = z0 + dz
    #             if 0 <= z < Z:
    #                 w = max(0.0, 1.0 - abs(dz) / float(radius)) # linear decay
    #                 slice_w[z] = max(slice_w[z], w)
    #     valid = np.maximum(valid, slice_w[:, None, None])
    
    # valid[roi] = 1.0

    # dis = distance_transform_edt(~roi)
    # safe_bg = (dis > safe_dis) & ~roi

    # valid[safe_bg] = 1.0 # safe negative
    # label[safe_bg] = 0.0
    
    return label, valid


class zarr3Ddataset(Dataset):
    def __init__(self, zarr_path):
        self.zarr_file = zarr.open(zarr_path, mode='r')
        self.keys = list(self.zarr_file.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        grp = self.zarr_file[key]

        mask3d = grp["mask"] # (S, H, W)
        raw = grp["raw"]   # (S, H, W)

        label, valid = make_label_valid(mask3d=mask3d)

        label = np.expand_dims(label, axis=0)
        valid = np.expand_dims(valid, axis=0)
        raw = np.expand_dims(raw, axis=0)

        return {"image": raw, "label":label, "valid":valid}

D, PH, PW = 32, 192, 192

trfm = Compose([
    ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    DivisiblePadd(keys=["image"], k=16, mode="constant"),
    ToTensord(keys=["image"]),
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
):
    chk = torch.load(checkpoint_filename, map_location=device)
    student_state_dict = chk["student_state_dict"]
    student = build_unet().to(device)
    student.load_state_dict(student_state_dict, strict=True)
    student.eval()

    ds = zarr3Ddataset(dataset_filename)
    test_ds = Dataset(ds, transform=trfm)
    loader = DataLoader(test_ds, shuffle=False, batch_size=1)
    zarr_f = zarr.open(save_filename, mode="a")

    for i, batch in enumerate(loader):
        x = batch["image"].to(device)
        x = x.float()
        logits = student(x)
        probs = torch.sigmoid(logits) if save_probs else logits

        x_np = x.detach().cpu().numpy()[0][0]       # (Z,H,W)
        y_np = probs.detach().cpu().numpy()[0][0]   # (Z,H,W)

        g = zarr_f.require_group(str(i))
        g.create_dataset("image", data=x_np, shape=x_np.shape, overwrite=True)
        g.create_dataset("pred", data=y_np, shape=y_np.shape, overwrite=True)
        
        if do_plot:
            z = plot_slice
            if z is None:
                z = x_np.shape[0] // 2  # middle slice

            img2d = x_np[z]          # (H,W)
            pred2d = y_np[z]         # (H,W)
            pred2d = (pred2d > 0.9).astype("int")

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img2d, cmap="gray")
            ax[0].set_title(f"Image | case {i} | slice {z}")
            ax[0].axis("off")

            ax[1].imshow(img2d, cmap="gray")
            ax[1].imshow(pred2d, cmap="jet", alpha=0.5)
            ax[1].set_title("Prediction overlay")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()

if __name__== "__main__":
    checkpoint_filename = r"C:\Users\bmahsa\Downloads\In Vivo Image_Wholemount (CART Samples)\train checkpoints\best_model.pth"
    dataset_filename = r"C:\Users\bmahsa\Downloads\Best Images + ROIs\test_dataset.zarr"
    save_filename = r"C:\Users\bmahsa\Downloads\Best Images + ROIs\test_segmentation.zarr"

    validate_model(checkpoint_filename, dataset_filename, save_filename, save_probs=True)