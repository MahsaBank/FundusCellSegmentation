import numpy as np
import zarr
import math
import torch
import copy
import torch.nn.functional as F
from monai.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt
from monai.transforms import (
    Compose, SpatialPadd, ScaleIntensityRangePercentilesd,
    RandCropByPosNegLabeld, RandFlipd, RandGaussianNoised, ToTensord, AdjustContrastd
)
from monai.networks.nets import UNet
from torch.optim import AdamW
import random
import os
import csv
import matplotlib.pyplot as plt
from torch.utils.data import random_split

def build_unet(in_channels=1, out_channels=1):
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

def make_label_valid(mask):
    mask = np.asarray(mask)
    label = (mask > 0).astype(np.float32)
    valid = (mask == 1).astype(np.float32)
    
    return label, valid


class zarr3Ddataset(Dataset):
    def __init__(self, zarr_path, selected_keys=None):
        self.zarr_file = zarr.open(zarr_path, mode='r')
        all_keys = list(self.zarr_file.keys())

        if selected_keys is None:
            self.keys = all_keys
        else:
            self.keys = [k for k in all_keys if k in selected_keys]

        self.index_map = []
        for k in self.keys:
            raw = np.asarray(self.zarr_file[k]["raw"])
            for i in range(raw.shape[0]):
                self.index_map.append((k, i))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key, i = self.index_map[idx]
        grp = self.zarr_file[key]

        mask2d = grp["mask"][i] # (H, W)
        raw2d = grp["raw"][i]   # (H, W)

        label, valid = make_label_valid(mask=mask2d)

        label = np.expand_dims(label, axis=0)
        valid = np.expand_dims(valid, axis=0)
        raw2d = np.expand_dims(raw2d, axis=0).astype(np.float32)

        return {"image": raw2d, "label":label, "valid":valid, "key":key, "slice_idx":i}


PH, PW = 192, 192

train_trfm = Compose([
    SpatialPadd(keys=["image", "label", "valid"], spatial_size=(PH, PW), mode="constant"),
    ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label", "valid"],
        label_key="label",
        spatial_size=(PH, PW),
        pos=0.7, neg=0.3,
        num_samples=4,
    ),
    # AdjustContrastd(
    #     keys="image",
    #     gamma=0.7,
    # ),
    # RandFlipd(keys=["image", "label", "valid"], prob=0.5, spatial_axis=0), # H fliping
    # RandFlipd(keys=["image", "label", "valid"], prob=0.5, spatial_axis=1), # W fliping
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.01),
    ToTensord(keys=["image", "label", "valid"]),
     ])

val_trfm = Compose([
    SpatialPadd(keys=["image", "label", "valid"], spatial_size=(PH, PW), mode="constant"),
    ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label", "valid"],
        label_key="label",
        spatial_size=(PH, PW),
        pos=0.7, neg=0.3,
        num_samples=4,
    ),
    # AdjustContrastd(
    #     keys="image",
    #     gamma=0.7,
    # ),
    RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.01),
    ToTensord(keys=["image", "label", "valid"]),
     ])


@torch.no_grad()
def ema_update(teacher, student, ema=0.99):
    for s, t in zip(student.parameters(), teacher.parameters()):
        t.data.mul_(ema).add_(s.data, alpha=1.0 - ema)

def sigmoid_rampup(current, rampup_length):
# @inproceedings{Tarvainen2017MeanTeacher,
#   title={Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results},
#   author={Tarvainen, Antti and Valpola, Harri},
#   booktitle={Advances in Neural Information Processing Systems},
#   volume={30},
#   year={2017}
# }
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


def masked_bce_dice_loss(logits, target, valid, eps=1e-6, bce_w=0.5):
    # valid weights between 0 and 1. 0 means unknown
    # size of inputs are (B, 1, H, W) 

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid).sum() / (valid.sum() + eps)

    prob = torch.sigmoid(logits)
    prob_w = prob * valid
    targ_w = target * valid
    inter = (prob_w * targ_w).sum()
    denom = prob_w.sum() + targ_w.sum() + eps
    dice = 1.0 - (2.0 * inter + eps) / denom

    return bce_w * bce + (1 - bce_w) * dice

def consistency_loss(student_logits, teacher_logits, valid, eps=1e-6):
    ps = torch.sigmoid(student_logits)
    pt = torch.sigmoid(teacher_logits)
    mse_map = (pt - ps) ** 2

    unsup = (valid == 0.0).float() 

    return (mse_map * unsup).sum() / (unsup.sum() + eps)

def train_mean_teacher(
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    device: str = "cuda",
    max_epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    ema: float = 0.99,
    unsup_max_weight: float = 1.0,
    unsup_rampup_iters: int = 2000,
    log_every: int = 20,
    checkpoint_path: str = "",
    available_checkpoint=None
):
    os.makedirs(checkpoint_path, exist_ok=True)
    bestmodel_filename = os.path.join(checkpoint_path, "best_model.pth")
    csv_file = os.path.join(checkpoint_path, "metrics.csv")

    # -----------------------------
    # CSV setup: append if exists
    # -----------------------------
    csv_exists = os.path.exists(csv_file)
    if not csv_exists:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "global_step",
                "train_loss_avg", "train_sup_avg", "train_cons_avg",
                "val_loss_avg",
                "lam", "valid_frac_avg"
            ])

    # -----------------------------
    # Build models
    # -----------------------------
    student = build_unet().to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # Resume state if checkpoint exists
    # -----------------------------
    start_epoch = 0
    global_step = 0
    best_score = float("inf")

    if available_checkpoint is not None and os.path.exists(available_checkpoint):
        chk = torch.load(available_checkpoint, map_location=device)

        student.load_state_dict(chk["student_state_dict"])

        if "teacher_state_dict" in chk:
            teacher.load_state_dict(chk["teacher_state_dict"])
        else:
            teacher = copy.deepcopy(student).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        if "optimizer_state_dict" in chk:
            opt.load_state_dict(chk["optimizer_state_dict"])

        start_epoch = chk.get("epoch", -1) + 1
        global_step = chk.get("global_step", 0)
        best_score = chk.get("best_score", float("inf"))

        print(
            f"[RESUME] Loaded checkpoint from {available_checkpoint} | "
            f"start_epoch={start_epoch}, global_step={global_step}, best_score={best_score:.4f}"
        )

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, start_epoch + max_epochs):
        student.train()

        run_loss = 0.0
        run_sup = 0.0
        run_cons = 0.0
        run_valid_frac = 0.0
        n_steps = 0
        lam = 0.0

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            v = batch["valid"].to(device)

            valid_frac = (v.sum() / (v.numel() + 1e-6)).item()
            run_valid_frac += valid_frac

            with torch.no_grad():
                t_logits = teacher(x)

            ramp = sigmoid_rampup(global_step, unsup_rampup_iters)
            lam = unsup_max_weight * ramp

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    s_logits = student(x)
                    L_sup = masked_bce_dice_loss(s_logits, y, v)
                    L_cons = consistency_loss(s_logits, t_logits, v)
                    loss = L_sup + lam * L_cons

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                s_logits = student(x)
                L_sup = masked_bce_dice_loss(s_logits, y, v)
                L_cons = consistency_loss(s_logits, t_logits, v)
                loss = L_sup + lam * L_cons

                loss.backward()
                opt.step()

            ema_update(teacher, student, ema=ema)

            run_loss += loss.item()
            run_sup += L_sup.item()
            run_cons += L_cons.item()
            n_steps += 1

            if (step % log_every) == 0:
                pos_vox = (y > 0).sum().item()
                print(
                    f"epoch {epoch:03d} step {step:04d} | "
                    f"loss={loss.item():.4f} sup={L_sup.item():.4f} cons={L_cons.item():.4f} | "
                    f"lam={lam:.3f} valid_frac={valid_frac:.3f} pos_vox={pos_vox}"
                )

            global_step += 1

        train_loss_avg = run_loss / max(n_steps, 1)
        train_sup_avg = run_sup / max(n_steps, 1)
        train_cons_avg = run_cons / max(n_steps, 1)
        valid_frac_avg = run_valid_frac / max(n_steps, 1)

        # -----------------------------
        # Validation
        # -----------------------------
        val_loss_avg = None
        if val_loader is not None:
            student.eval()
            vloss = 0.0
            vsteps = 0
            with torch.no_grad():
                for vb in val_loader:
                    vx = vb["image"].to(device)
                    vy = vb["label"].to(device)
                    vv = vb["valid"].to(device)
                    vlogits = student(vx)
                    L = masked_bce_dice_loss(vlogits, vy, vv)
                    vloss += L.item()
                    vsteps += 1
            val_loss_avg = vloss / max(vsteps, 1)
            print(f"[VAL] epoch {epoch:03d} loss={val_loss_avg:.4f}")

        # -----------------------------
        # Save checkpoint
        # -----------------------------
        score = val_loss_avg if val_loss_avg is not None else train_loss_avg

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "best_score": best_score,
        }

        ckpt_name = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, ckpt_name)

        if score < best_score:
            best_score = score
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, bestmodel_filename)
            print(f"[BEST] epoch {epoch:03d} score={best_score:.4f} -> saved best_model.pth")

        # -----------------------------
        # Append metrics
        # -----------------------------
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, global_step,
                train_loss_avg, train_sup_avg, train_cons_avg,
                val_loss_avg if val_loss_avg is not None else "",
                lam, valid_frac_avg
            ])

        print(
            f"[TRAIN] epoch {epoch:03d} avg | "
            f"loss={train_loss_avg:.4f} sup={train_sup_avg:.4f} cons={train_cons_avg:.4f} | "
            f"valid_frac_avg={valid_frac_avg:.3f}"
        )

    return student, teacher


dataset_filename = "/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/train.zarr"
zarr_file = zarr.open(dataset_filename, mode="r")
all_keys = list(zarr_file.keys())

random.seed(42)
all_keys = all_keys.copy()
random.shuffle(all_keys)

# 80/20 split by key
train_key_count = int(0.8 * len(all_keys))
train_keys = all_keys[:train_key_count]
val_keys = all_keys[train_key_count:]

print(f"Total keys: {len(all_keys)}")
print(f"Train keys: {len(train_keys)}")
print(f"Val keys: {len(val_keys)}")

train_base_dataset = zarr3Ddataset(dataset_filename, selected_keys=train_keys)
val_base_dataset = zarr3Ddataset(dataset_filename, selected_keys=val_keys)

train_dataset = Dataset(train_base_dataset, transform=train_trfm)
val_dataset = Dataset(val_base_dataset, transform=val_trfm)

loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=5, shuffle=False)

student, teacher = train_mean_teacher(
    train_loader=loader,
    val_loader=loader_val,
    device="cpu",
    max_epochs=2000,
    ema=0.99,
    unsup_max_weight=0.5,
    unsup_rampup_iters=2000,
    checkpoint_path="/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/training_2d_notFlattening_splitData",
    available_checkpoint="/storage2/fs1/leeay/Active/bmahsa/workplaces/CellFundusSegmentation/training_2d_notFlattening_splitData/best_model.pth"
)
