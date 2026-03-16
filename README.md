# prepare_cell_dataset.py

## Overview

`prepare_cell_dataset.py` builds a **Zarr dataset for cell segmentation training and testing** from:

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


PreR_AllSamples/
├── sample1.zip
├── sample1.tif
├── sample2.zip
├── sample2.tif


If ROI files start with `RoiSet`, the script automatically resolves the matching TIFF filename.

---

## Output Dataset

The script generates a Zarr dataset such as:


train.zarr
├── sample1
│ ├── raw (Z, H, W)
│ └── mask (Z, H, W)
├── sample2
│ ├── raw
│ └── mask


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
raw3d = img[1, :, :, :]

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
Required libraries:
```python
numpy

matplotlib

scipy

tifffile

zarr

read-roi
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
