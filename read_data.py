from read_roi import read_roi_zip
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from matplotlib.animation import FuncAnimation
from scipy.ndimage import rotate
import tifffile as tiff
import zarr


def _crop_patch(img2d, cy, cx, half):
    H, W = img2d.shape
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    patch = img2d[y0:y1, x0:x1]
    return patch

def _hist_feat(patch, bins=32, eps=1e-8):
    # robust histogram on patch intensities
    p = patch.astype(np.float32)
    vmin, vmax = float(np.min(p)), float(np.max(p))
    if vmax <= vmin + eps:
        # flat patch
        h = np.zeros((bins,), np.float32)
        h[0] = 1.0
        return h
    h, _ = np.histogram(p, bins=bins, range=(vmin, vmax))
    h = h.astype(np.float32)
    h /= (h.sum() + eps)
    return h

def _cosine(a, b, eps=1e-8):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def add_ellipse(mask2d, top, left, h, w, value=1):
    H, W = mask2d.shape

    y0 = max(0, int(top))
    x0 = max(0, int(left))
    y1 = min(H, y0 + int(h))
    x1 = min(W, x0 + int(w))

    if x0 >= x1 or y0 >= y1:
        return

    cx = (x0 + x1 - 1) / 2.0
    cy = (y0 + y1 - 1) / 2.0
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    if rx <= 0 or ry <= 0:
        return

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    X, Y = np.meshgrid(xs, ys)

    inside = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0

    patch = mask2d[y0:y1, x0:x1]   
    patch[inside] = value          

def propagate_cell_labels_by_hist(
    raw3d,                 # (Z,H,W) float or uint
    label3d,               # (Z,H,W) binary, contains initial ovals in some slices
    cells,                 # list of ROI dicts from read_roi_zip values (each has top,left,height,width, position['slice'])
    radius=5,
    patch_half=12,         # patch size = (2*patch_half+1)
    search_xy=4,           # search shifts in [-search_xy, +search_xy]
    bins=32,
    sim_thr=0.92,          # histogram cosine similarity threshold
    mean_rel_thr=0.20,     # |mean2-mean1| / (|mean1|+eps)
    std_rel_thr=0.40,      # same for std
    min_new_area=20,       # reject tiny labels
    add_ellipse_fn=add_ellipse,   # function(mask2d, top,left,h,w,value=1)
    eps=1e-8,
):
    """
    Returns updated_label3d (binary) with propagated labels.
    Strategy:
      For each cell labeled in slice z0:
        - compute reference patch around its center in raw3d[z0]
        - for each z in [z0-radius, z0+radius], z!=z0:
            search small (dx,dy) shifts; take best similarity
            if similarity & stats pass => add ellipse label in that slice
    """
    assert raw3d.ndim == 3 and label3d.ndim == 3
    Z, H, W = raw3d.shape
    out = (label3d > 0).astype(np.uint8).copy()

    if add_ellipse_fn is None:
        raise ValueError("Pass add_ellipse_fn=add_ellipse (your function)")

    # precompute raw as float for stable stats
    raw = raw3d.astype(np.float32)

    for c in cells:
        z0 = int(c["position"]["slice"])
        if not (0 <= z0 < Z):
            continue

        top = float(c["top"]); left = float(c["left"])
        hh = float(c["height"]); ww = float(c["width"])

        # center of ellipse/bbox in image coords
        cy0 = int(round(top + hh / 2.0))
        cx0 = int(round(left + ww / 2.0))

        # reference patch at z0
        ref_patch = _crop_patch(raw[z0], cy0, cx0, patch_half)
        ref_hist = _hist_feat(ref_patch, bins=bins)
        ref_mean = float(ref_patch.mean())
        ref_std = float(ref_patch.std())

        # search neighboring slices
        for dz in range(-radius, radius + 1):
            if dz == 0:
                continue
            z = z0 + dz
            if not (0 <= z < Z):
                continue

            best = -1.0
            best_xy = (0, 0)
            best_stats = None

            # small xy search (optional drift)
            for dy in range(-search_xy, search_xy + 1):
                for dx in range(-search_xy, search_xy + 1):
                    cy = cy0 + dy
                    cx = cx0 + dx
                    if not (0 <= cy < H and 0 <= cx < W):
                        continue

                    patch = _crop_patch(raw[z], cy, cx, patch_half)
                    hfeat = _hist_feat(patch, bins=bins)
                    sim = _cosine(ref_hist, hfeat)

                    if sim > best:
                        m = float(patch.mean())
                        s = float(patch.std())
                        best = sim
                        best_xy = (dy, dx)
                        best_stats = (m, s)

            if best_stats is None:
                continue

            m2, s2 = best_stats
            mean_ok = (abs(m2 - ref_mean) / (abs(ref_mean) + eps)) <= mean_rel_thr
            std_ok  = (abs(s2 - ref_std) / (abs(ref_std) + eps)) <= std_rel_thr

            if (best >= sim_thr) and mean_ok and std_ok:
                dy, dx = best_xy

                # place ellipse at shifted position in slice z
                top2 = top + dy
                left2 = left + dx

                before = out[z].sum()
                add_ellipse_fn(out[z], top2, left2, hh, ww, value=10)
                after = out[z].sum()

                # reject tiny additions
                if (after - before) < min_new_area:
                    # undo if it added almost nothing
                    out[z] = out[z].copy()
                    pass

    return out

def build_union_mask(cells, mask):
    for c in cells:
        top = c["top"]       # y
        left = c["left"]     # x
        h = c["height"]
        w = c["width"]

        f = int(c["position"]["frame"])
        s = int(c["position"]["slice"])
        ch = int(c["position"]["channel"])

        if not (0 <= f < mask.shape[0] and 0 <= s < mask.shape[1] and 0 <= ch < mask.shape[4]):
            continue

        mask2d = mask[f, s, :, :, ch]
        add_ellipse(mask2d, top, left, h, w)


def build_dataset(folder, zarr_name="training_dataset.zarr"):
    fnames = glob.glob(os.path.join(folder, "*.zip"))
    zarr_filename = os.path.join(os.path.dirname(folder), zarr_name)
    zarr_file = zarr.open(zarr_filename, mode='w')
    dataset = {}

    # selected_sample = "RoiSet_163-0 RE pre top" 
    for fn in fnames:
        base = os.path.basename(fn).split(".zip")[0]

        # if base != selected_sample:
        #     continue
        
        print(base)
        data = read_roi_zip(fn)

        if "RoiSet" not in base:
            img_filename = os.path.join(os.path.dirname(fn), base + ".tif")
            name = base
        else:
            img_filename = os.path.join(os.path.dirname(fn), base.split("RoiSet")[1][1:] + ".tif")
            name = base.split("RoiSet")[1][1:]

        if not os.path.exists(img_filename):
            continue

        img = np.asarray(tiff.imread(img_filename))
        img = img.transpose((1, 0, 2, 3))
        H, W = img.shape[2], img.shape[3]
        T, S, C = img.shape[0], img.shape[1], 1 
        mask = np.zeros((T, S, H, W, C), dtype=np.uint8)
        cells = list(data.values())
        build_union_mask(cells, mask)

        raw3d = img[1, :, :, :] # frame 1 has information
        mask3d = mask[1, :, :, :, 0]

        new_mask3d = propagate_cell_labels_by_hist(
            raw3d=raw3d,
            label3d=mask3d,
            cells=cells,
            radius=10,
            patch_half=10,
            search_xy=3,
            bins=32,
            sim_thr=0.93,
            mean_rel_thr=0.25,
            std_rel_thr=0.50,
            add_ellipse_fn=add_ellipse,
            )

        dataset[name] = {"raw": raw3d, "mask": new_mask3d}
        bgroup = zarr_file.create_group(name)
        bgroup.create_dataset("raw", data=raw3d, shape=raw3d.shape)
        bgroup.create_dataset("mask", data=new_mask3d, shape=new_mask3d.shape)
        print("mask sum:", new_mask3d.sum(), "mask max:", new_mask3d.max())

    print("finished building dataset. items:", len(dataset))


folder = r"C:\Users\bmahsa\Downloads\PreR_AllSamples\PreR_AllSamples"
build_dataset(folder=folder, zarr_name="train.zarr")

zarr_filename = r"C:\Users\bmahsa\Downloads\PreR_AllSamples\train.zarr"
dataset = zarr.open(zarr_filename, mode='r')
idx = 0
key = list(dataset.keys())[idx]
img = dataset[key]["raw"]          # (S,512,512)
mask = dataset[key]["mask"]        # (S,512,512)

f, ch = 1, 0
S = mask.shape[0]

if img.ndim == 2:  
    img_stack = np.repeat(img[None, :, :], S, axis=0)   # (S,512,512)
elif img.ndim >= 3: 
    img_stack = img
else:
    raise ValueError(f"Unexpected img shape: {img.shape}")

def mask_to_rgb(mask2d):
    rgb = np.zeros((*mask2d.shape, 3), dtype=np.uint8)
    rgb[mask2d == 1] = (255, 0, 0)   # original label = red
    rgb[mask2d == 10] = (0, 255, 0)  # propagated label = green
    return rgb

fig, ax = plt.subplots(figsize=(6, 6))

im0 = ax.imshow(img_stack[0], cmap="gray", origin="upper")
im1 = ax.imshow(mask_to_rgb(mask[0]), alpha=0.3, origin="upper") 

ax.axis("off")

def update(s):
    im0.set_data(img_stack[s])
    im1.set_data(mask_to_rgb(mask[s]))
    ax.set_title(f"slice={s} | mask_sum={int(mask[s].sum())}")
    return im0, im1

ani = FuncAnimation(fig, update, frames=S, interval=200, blit=False)
plt.show()

# prevent Animation.__del__ shutdown warning
try:
    ani.event_source.stop()
except Exception:
    pass
plt.close(fig)
del ani
