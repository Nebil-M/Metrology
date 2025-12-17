# Robust SEM Masking: Iterative Training with U-Net + Evaluator

## Overview

SEM images of wafer cross-sections are difficult to segment reliably: **classical binarization + morphology fails** across many structures and contrast conditions, while **manual mask curation is accurate but too slow to scale**.

We implemented:

- **U-Net** proposes a segmentation mask for a raw SEM image
- **Evaluator** scores the proposed mask as **Good / Bad**
- **Iterative training** grows the labeled dataset by adding only **high-confidence (Good) pseudo-labels**, then retrains the U-Net in stages

---

## Method Summary

### Iterative training loop

1. Train U-Net on the current labeled dataset (`image/`, `label/`)
2. Run U-Net inference on a large unlabeled SEM set → predicted masks
3. Evaluator classifies each (image, predicted mask) as **Good** or **Bad**
4. Keep only **Good** predictions and build the next-stage dataset
5. Repeat for several stages, growing the training set each time

---

## Project Layout (core pieces)

```text
code/
  unet/              # U-Net model + training + inference
  rnn_quality/       # (if present) quality-model utilities
  Evaluator.py       # evaluator utilities
  Util.py            # manager/wrapper helpers
data/
  <stage_k>/
    image/           # *.tif grayscale SEM images
    label/           # *.tif (or *.png) binary masks with matching stems
models/
  unet_stage_k.pth
  evaluator.pth
```

> Notes:
>
> - U-Net training assumes `data_root/image/*.tif` + `data_root/label/*.tif` with matching filename stems.
> - Datasets for iterative stages are created automatically by the iterative training script.

---

## Evaluator (Good/Bad Mask Classifier)

### Task definition

Input: **(raw SEM image, candidate mask)** → label ∈ {**Good**, **Bad**}

### Dataset construction

- **Good** set: trusted masks from Shuhan’s labeling (curated)
- **Bad** set: noisy/incorrect masks from labeling
- Scripts build:
  - `1_Good/` and `0_Bad/`
  - each with aligned `images/` and `masks/`

### Model

- **ResNet18**
- First convolution modified to accept **4 channels**: **RGB + mask**
- Final fully-connected layer outputs **one logit** = P(mask is Good)

### Training details

- Train/val split: **80/20**
- Batch size: **16**
- Learning rate: **1e-4**
- ~**40 epochs**
- Loss: **BCEWithLogitsLoss** with `pos_weight` from class counts (handles imbalance)

**Selected decision threshold:** **0.7392**

---

## U-Net (SEM Mask Proposer)

### Architecture

- Custom U-Net (`unet.Unet`)
- Channels: **(32, 64, 128, 256)**
- Input: **1-channel grayscale SEM**
- Output: **1-channel probability mask**

### Training details

- Loss: **BCEDiceLoss** (BCEWithLogits + Dice)
- Optimizer: **Adam**
- Learning rate: **1e-4**
- Batch size: **8**
- ~**30 epochs per stage**
- Train/val split: **80/20 random split**

### Dataset notes from the presentation

- TCL and FLG were not used in one training pass due to **duplicates in the data**
- TTGAP was not found in Shuhan’s curation
- Reported: U-Net generalizes better across structures than the evaluator (evaluator is more structure-sensitive)

---

## How to Run Iterative Training

The full iterative training loop is handled by **`iterative_training.py`**, which performs:
- U-Net training
- inference on unlabeled data
- evaluator scoring
- automatic construction of the next-stage dataset

### 0) Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install numpy opencv-python torch matplotlib scikit-image tifffile
```

### 1) Configure paths and parameters

Edit the `__main__` block in `iterative_training.py`:

```python
BASE_DATA_ROOT = "Iterative_dataset\\initial_dataset"
UNLABELED_ROOT = "Iterative_dataset\\True_RAW_Data_flattened"
WORK_ROOT = "Iterative_dataset\\iterative_runs"
EVALUATOR_MODEL = "Iterative_dataset\\Balanced_evalutor.pth"

THRESHOLD = 0.7392
```

### 2) Run the iterative loop

```bash
python iterative_training.py
```

### 3) Outputs

During execution, the script automatically generates:

- `unet_stage{k}.pth` — U-Net checkpoint at each stage
- `preds_stage{k}/` — predicted masks on the unlabeled dataset
- `data_stage{k+1}/image/` and `data_stage{k+1}/label/` — the next-stage dataset built from:
  - all previous stage data, plus
  - evaluator-accepted pseudo-labels
- `stage{k}_stats.json` — evaluator probabilities and threshold used

---

## Recommended Experiment Logging

Track, per stage:

- Number of images added by evaluator
- Evaluator score distribution and acceptance rate at threshold τ
- U-Net validation loss / Dice (or IoU) per stage
- Qualitative samples: accepted vs rejected masks (for failure mode analysis)

---

## Dependencies

- PyTorch
- NumPy
- OpenCV
- scikit-image
- Matplotlib
- (optional) tifffile for robust TIFF I/O

---

## License

MIT License
