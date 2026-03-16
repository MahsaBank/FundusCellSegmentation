# prepare_cell_dataset.py

## Overview

`prepare_cell_dataset.py` builds a **Zarr dataset for cell segmentation training** from:

- TIFF image stacks (`.tif`)
- ImageJ/Fiji ROI annotation files (`.zip`)

The script converts manually annotated cell ROIs into segmentation masks and **propagates those labels to nearby slices** using histogram-based patch similarity. This produces a **denser 3D annotation volume** suitable for training deep learning models.

The final dataset is saved in **Zarr format**, containing raw images and generated masks.

---

## Input Data

The script expects a folder containing:

- ROI annotation files (`*.zip`)
- Corresponding TIFF image stacks (`*.tif`)

Example structure:
```
PreR_AllSamples/
├── sample1.zip
├── sample1.tif
├── sample2.zip
├── sample2.tif
```

If ROI files start with `RoiSet`, the script automatically resolves the matching TIFF filename.

---

## Output Dataset

The script generates a Zarr dataset such as:

```
train.zarr
├── sample1
│ ├── raw (Z, H, W)
│ └── mask (Z, H, W)
├── sample2
│ ├── raw
│ └── mask
```

### Stored arrays

| Dataset | Description |
|--------|-------------|
| `raw` | 3D grayscale image volume |
| `mask` | 3D segmentation mask |

### Mask label values

| Value | Meaning |
|------|--------|
| 0 | background |
| 1 | manually annotated cells |
| 10 | propagated cell labels |

---

## Label Propagation Method

The function `propagate_cell_labels_by_hist()` extends manually labeled cells into neighboring slices.

### Algorithm

1. Extract a **reference patch** around the cell center in the labeled slice.
2. Search neighboring slices within a specified radius.
3. Compare patches using:
   - histogram cosine similarity
   - relative mean intensity difference
   - relative standard deviation difference
4. If similarity conditions are satisfied, an ellipse label is added to the neighboring slice.

This allows **sparse manual annotations to be expanded into a denser 3D training mask**.

---

## Main Functions

| Function | Description |
|--------|-------------|
| `_crop_patch(img2d, cy, cx, half)` | Extracts a square patch centered at `(cy, cx)` |
| `_hist_feat(patch)` | Computes a normalized histogram feature vector |
| `_cosine(a, b)` | Computes cosine similarity between histogram vectors |
| `add_ellipse(mask2d, top, left, h, w)` | Draws an ellipse ROI into a mask |
| `build_union_mask(cells, mask)` | Builds the initial mask volume from ROI annotations |
| `propagate_cell_labels_by_hist(...)` | Propagates cell labels to neighboring slices |
| `build_dataset(folder, zarr_name)` | Processes all ROI/TIFF pairs and builds the Zarr dataset |

---

## Image Format

After loading TIFF stacks, the script assumes images are arranged as:


(T, S, H, W)


Where:

- **T** → frame  
- **S** → slice  
- **H** → image height  
- **W** → image width  

The script currently selects **frame 1** for dataset creation:
```python
raw3d = img[1, :, :, :]
```
## Visualization

After building the dataset, the script loads one sample and displays an animated viewer showing image slices with mask overlays.

### Color Coding

| Color | Meaning |
|------|--------|
| Red | original manual labels |
| Green | propagated labels |

This visualization helps verify that label propagation is working correctly.

---

## Adjustable Parameters

Inside `propagate_cell_labels_by_hist()`:

| Parameter | Description |
|----------|-------------|
| radius | slice search range |
| patch_half | patch half-size |
| search_xy | allowed XY shift |
| bins | histogram bins |
| sim_thr | cosine similarity threshold |
| mean_rel_thr | relative mean difference threshold |
| std_rel_thr | relative std difference threshold |
| min_new_area | minimum accepted propagated area |

### Example values used

```python
radius = 10
patch_half = 10
search_xy = 3
bins = 32
sim_thr = 0.93
mean_rel_thr = 0.25
std_rel_thr = 0.50
```

## Running the Script

Modify the folder path near the end of the script:
```python
folder = r"C:\Users\bmahsa\Downloads\PreR_AllSamples\PreR_AllSamples"
```
The output dataset name is controlled by the variable zarr_name:
```python
build_dataset(folder=folder, zarr_name="train.zarr")
```
Then run:
```python
python prepare_cell_dataset.py
```
The script will create:
```python
train.zarr
```
in the parent directory.

## Requirements

Install required packages:
```python
pip install numpy matplotlib scipy tifffile zarr read-roi
```
## Notes

- Propagated labels provide additional supervision but are not equivalent to manual annotations.

- Propagation quality depends on image focus and slice-to-slice consistency.

- The method works best when cells appear similar across neighboring slices.

## Use Case

This script is intended for preparing segmentation datasets when:

- only a subset of cells are manually labeled

- labels need to be extended across slices

- training data must be generated for deep learning segmentation models.

---

# train_model_v1.py

## Overview

`train_model_v1.py` trains a **cell segmentation model** using a **Mean Teacher semi-supervised learning framework** implemented with **PyTorch and MONAI**.

The training pipeline uses:

- manually annotated cells as **supervised labels**
- propagated labels from `read_data.py` as **unlabeled regions**
- a **student–teacher model architecture** to enforce consistency between predictions.

The script trains a **2D U-Net segmentation network** using flattened projections of 3D volumes.

---

## Training Strategy

The training uses the **Mean Teacher framework**:

1. A **student network** is trained using labeled pixels.
2. A **teacher network** is maintained as an exponential moving average (EMA) of the student weights.
3. The teacher generates stable predictions used for **consistency regularization**.
4. The student is optimized using a combination of:

- supervised segmentation loss
- unsupervised consistency loss

Reference:

Tarvainen & Valpola, *Mean teachers are better role models*, NeurIPS 2017.

---

## Dataset Format

The script expects a dataset generated by `prepare_cell_dataset.py`:

```
train.zarr
├── sample1
│ ├── raw (Z, H, W)
│ └── mask (Z, H, W)
```

### Mask values

| Value | Meaning |
|------|--------|
| 0 | background |
| 1 | manually labeled cells |
| >1 | propagated cell annotations |

During training:

- **manual labels (1)** are used for supervised loss
- **propagated labels (>1)** are treated as **unknown regions**

---

## Data Preprocessing

3D image volumes are converted into **2D representations** using maximum intensity projections:

```python
raw_2D = max(raw3d, axis=0)
label_2D = max(label3d, axis=0)
valid_2D = max(valid3d, axis=0)
```

Where:

- `label` → binary segmentation target
- `valid` → mask indicating supervised regions.

---

## Data Augmentation

Training transformations are implemented using **MONAI transforms**:

- spatial padding
- intensity normalization
- positive/negative patch sampling
- random flips
- Gaussian noise

Patch size:

192 × 192


Example augmentations:

- `RandCropByPosNegLabeld`
- `RandFlipd`
- `RandGaussianNoised`

---

## Loss Functions

### Supervised Loss

Combination of:

- **Binary Cross Entropy**
- **Dice Loss**

Computed only on **valid labeled pixels**.

```python
L_sup = BCE + Dice
```

### Consistency Loss

Mean squared error between:

- student predictions
- teacher predictions

Applied only on **unlabeled pixels**.

```python
L_cons = MSE(student, teacher)
```

### Final Loss

```python
L = L_sup + λ * L_cons
```

Where λ gradually increases during training using a **sigmoid ramp-up schedule**.

---

## Model Architecture

The segmentation network is a **2D U-Net** implemented using MONAI.

Encoder channel sizes:

```python
(32, 64, 128, 256, 512)
```

Strides:

```python
(2, 2, 2, 2)
```

Residual units:

```python
num_res_units = 2
```

---

## Teacher Model Update

The teacher network parameters are updated using **exponential moving average**:

```python
teacher = ema * teacher + (1 - ema) * student
```

Default EMA value:

```python
ema = 0.99
```

---

## Training Outputs

Training generates the following files:

```
training_2d/
├── best_model.pth
├── checkpoint_epoch_XXX.pth
├── metrics.csv
```

### Files

| File | Description |
|----|----|
| `best_model.pth` | best performing model |
| `checkpoint_epoch_*.pth` | saved checkpoints |
| `metrics.csv` | training statistics |

---

## Training Metrics

The CSV log includes:

| Metric | Description |
|------|------|
| epoch | training epoch |
| global_step | total iterations |
| train_loss_avg | total training loss |
| train_sup_avg | supervised loss |
| train_cons_avg | consistency loss |
| val_loss_avg | validation loss |
| lam | unsupervised loss weight |
| valid_frac_avg | fraction of supervised pixels |

---

## Running the Training Script

Modify dataset and checkpoint paths:

```python
dataset_filename = "train.zarr"
checkpoint_path = "training_2d"
```
Example execution:
```
python train_model_v1.py
```
Example training configuration:
```python
student, teacher = train_mean_teacher(
    train_loader=loader,
    device="cpu",
    max_epochs=300,
    ema=0.99,
    unsup_max_weight=0.5,
    unsup_rampup_iters=2000,
)
```
## Key Hyperparameters

| Parameter | Description |
|-----------|-------------|
| max_epochs | number of training epochs |
| lr | learning rate |
| ema | teacher update rate |
| unsup_max_weight | max consistency weight |
| unsup_rampup_iters | ramp-up iterations |
| batch_size | training batch size |

## Notes

- Training uses semi-supervised learning to leverage sparse annotations.

- Only manually labeled cells contribute to supervised loss.

- Propagated labels are used only to define unlabeled regions.

- The approach helps improve training stability when manual annotations are limited.

## Use Case

This training pipeline is designed for:

- cell segmentation tasks

- datasets with limited manual annotations

- semi-supervised learning scenarios

- microscopy image analysis workflows.
