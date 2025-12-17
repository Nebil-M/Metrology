import os
import json
import shutil
from pathlib import Path
from typing import Literal, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from unet_training import train_stage
from unet_inference import infer_all


def convert_path(path: str) -> str:
    return path.replace("\\", "/") if "\\" in path else path


# ---------- EVALUATOR DATASET & MODEL ----------

class EvaluatorInferenceDataset(Dataset):
    """
    Dataset for running the evaluator on predicted masks.
    """
    def __init__(self, image_root: str, mask_root: str):
        self.samples: List[tuple[str, str, str]] = []
        valid_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

        # Map masks by stem
        masks_by_stem: Dict[str, str] = {}
        for root, _, files in os.walk(mask_root):
            for f in files:
                if f.lower().endswith(valid_exts):
                    stem = os.path.splitext(f)[0]
                    masks_by_stem[stem] = os.path.join(root, f)

        # Match images to masks by stem
        for root, _, files in os.walk(image_root):
            for f in files:
                if f.lower().endswith(valid_exts):
                    stem = os.path.splitext(f)[0]
                    if stem in masks_by_stem:
                        img_path = os.path.join(root, f)
                        msk_path = masks_by_stem[stem]
                        self.samples.append((img_path, msk_path, f))

        print(f"[Evaluator] Found {len(self.samples)} image/mask pairs to classify.")
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, fname = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.resize(image)
        mask = self.resize(mask)

        img_tensor = self.to_tensor(image)   # (3, H, W)
        msk_tensor = self.to_tensor(mask)    # (1, H, W)
        inp = torch.cat([img_tensor, msk_tensor], dim=0)  # (4, H, W)

        return inp, fname


def get_evaluator_model() -> nn.Module:
    model = models.resnet18(weights=None)
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = new_conv
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model


def classify_good_bad(
    image_dir: str,
    mask_dir: str,
    evaluator_model_path: str,
    device: Literal["cpu", "cuda"] = "cuda",
    batch_size: int = 16,
    threshold: float = 0.9, 
) -> Dict[str, int]:
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = EvaluatorInferenceDataset(image_dir, mask_dir)
    if len(dataset) == 0:
        print("[Evaluator] No samples to classify. Returning empty dict.")
        return {}, {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not os.path.exists(evaluator_model_path):
        raise FileNotFoundError(f"[Evaluator] Model not found at {evaluator_model_path}")

    model = get_evaluator_model().to(device_obj)
    try:
        state = torch.load(evaluator_model_path, map_location=device_obj)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Error loading evaluator model: {e}")
        raise
        
    model.eval()

    labels: Dict[str, int] = {}
    raw_probs: Dict[str, float] = {}

    with torch.no_grad():
        for inputs, fnames in loader:
            inputs = inputs.to(device_obj)
            logits = model(inputs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for fname, p in zip(fnames, probs):
                is_good = int(p >= threshold)
                labels[fname] = is_good
                raw_probs[fname] = float(p)

    print("[Evaluator] Finished classifying masks as Good/Bad.")
    return labels, raw_probs


# ---------- UNION DATASET BUILDER ----------

def build_union_dataset(
    prev_data_root: Path,
    unlabeled_images_root: Path,
    preds_root: Path,
    keep_names: List[str],
    out_data_root: Path,
) -> None:
    """
    Creates a new dataset folder that combines:
    1. All data from prev_data_root (Training set history)
    2. The 'keep_names' from the unlabeled set (New "Good" pseudo-labels)
    """
    out_img_dir = out_data_root / "image"
    out_lbl_dir = out_data_root / "label"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset] Building union dataset at {out_data_root}...")

    # 1. Copy everything from previous dataset
    # Assuming prev_data_root has /image and /label structure
    src_prev_img = prev_data_root / "image"
    src_prev_lbl = prev_data_root / "label"
    
    # Some datasets call it "images" vs "image" or "masks" vs "label"
    if not src_prev_img.exists():
        src_prev_img = prev_data_root / "images"
    if not src_prev_lbl.exists():
        src_prev_lbl = prev_data_root / "masks"

    count_prev = 0
    if src_prev_img.exists():
        for f in os.listdir(src_prev_img):
            src_file = src_prev_img / f
            if src_file.is_file():
                shutil.copy2(src_file, out_img_dir / f)
                # Try to copy matching label
                # Labels might have same name or slightly different extension
                # Simplest assumption: same filename
                if (src_prev_lbl / f).exists():
                    shutil.copy2(src_prev_lbl / f, out_lbl_dir / f)
                count_prev += 1
    else:
        print(f"[Dataset] Warning: Could not find image folder in {prev_data_root}")
        
    print(f"[Dataset] Copied {count_prev} samples from previous stage.")

    # 2. Copy new accepted pseudo-labels
    count_new = 0
    
    # We need to find the source image path for the keep_names
    # Since filenames might be duplicated in subfolders of unlabeled_root, 
    # we do a quick walk to map names to full paths
    unlabeled_map = {}
    for root, _, files in os.walk(unlabeled_images_root):
        for f in files:
            unlabeled_map[f] = Path(root) / f

    for fname in keep_names:
        # 2a. Copy Mask (Predicted)
        
        img_stem = os.path.splitext(fname)[0]
        mask_src = None
        for ext in ['.png', '.tif', '.tiff', '.jpg']:
            potential_mask = preds_root / f"{img_stem}{ext}"
            if potential_mask.exists():
                mask_src = potential_mask
                break
        
        if mask_src and fname in unlabeled_map:
            # Copy Image to new dataset
            shutil.copy2(unlabeled_map[fname], out_img_dir / fname)
            
            # Copy Mask to new dataset
            # Force the mask filename to match the image filename stem + mask extension
            dst_mask_name = mask_src.name
            
            # If your dataset loader requires mask name == image name (minus extension),
            # ensure that logic holds.
            shutil.copy2(mask_src, out_lbl_dir / dst_mask_name)
            count_new += 1
            
    print(f"[Dataset] Added {count_new} new high-confidence samples.")
    print(f"[Dataset] Total size: {count_prev + count_new}")


# ---------- ITERATIVE TRAINING LOOP ----------

def iterative_unet_training(
    base_data_root: str,
    unlabeled_images_root: str,
    work_root: str,
    evaluator_model_path: str,
    n_stages: int = 2,
    keep_threshold: float = 0.6,
    device: Literal["cpu", "cuda"] = "cuda",
    channels=(32, 64, 128, 256),
    epochs: int = 50,
    lr: float = 1e-4,
    bs: int = 8,
) -> List[Path]:
    
    base_data_root = Path(convert_path(base_data_root)).resolve()
    unlabeled_images_root = Path(convert_path(unlabeled_images_root)).resolve()
    work_root = Path(convert_path(work_root)).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"[Iterative] Base dataset:   {base_data_root}")
    print(f"[Iterative] Unlabeled root: {unlabeled_images_root}")
    print(f"[Iterative] Work root:      {work_root}")
    print(f"[Iterative] Stages: {n_stages}, keep_threshold={keep_threshold}")

    # Check input structure (looking for 'images'/'image' and 'masks'/'label')
    # Relaxing check to see if ANY image folder exists
    if not ((base_data_root / "image").exists() or (base_data_root / "images").exists()):
        print(f"Error: {base_data_root}/image (or images) not found.")
        return []

    current_data_root = base_data_root
    model_paths: List[Path] = []

    for stage in range(1, n_stages + 1):
        print("\n" + "=" * 80)
        print(f"[Iterative] STAGE {stage}/{n_stages}")
        print("=" * 80)

        # 1) Train U-Net
        model_pth = work_root / f"unet_stage{stage}.pth"
        print(f"[Iterative] Training stage-{stage} model...")

        train_stage(
            data_root=str(current_data_root),
            model_output=str(model_pth),
            epochs=epochs,
            lr=lr,
            bs=bs,
            device=device,
            channels=channels,
        )
        model_paths.append(model_pth)

        if stage == n_stages:
            print(f"[Iterative] Final stage completed.")
            break

        # 2) Inference
        preds_root = work_root / f"preds_stage{stage}"
        print(f"[Iterative] Inferring on unlabeled data...")
        infer_all(
            model_pth=str(model_pth),
            images_root=str(unlabeled_images_root),
            save_root=str(preds_root),
            device=device,
            channels=channels,
            thresh=0.5, # Binary threshold for the mask itself
        )

        # 3) Evaluate & Filter
        print(f"[Iterative] Evaluating predictions...")
        good_bad_labels, raw_probs = classify_good_bad(
            image_dir=str(unlabeled_images_root),
            mask_dir=str(preds_root),
            evaluator_model_path=evaluator_model_path,
            device=device,
            batch_size=16,
            threshold=keep_threshold,
        )

        if not good_bad_labels:
            print("[Iterative] No valid predictions found. Stopping loop.")
            break

        # Save stats
        debug_json = work_root / f"stage{stage}_stats.json"
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump({"probs": raw_probs, "threshold": keep_threshold}, f, indent=2)

        # Filter
        keep_names = [name for name, label in good_bad_labels.items() if label == 1]
        print(f"[Iterative] Evaluator accepted {len(keep_names)} / {len(good_bad_labels)} samples.")

        if len(keep_names) == 0:
            print("[Iterative] Warning: 0 samples passed the threshold. Next stage will effectively retry the same training.")
        
        # 4) Build Next Dataset
        next_data_root = work_root / f"data_stage{stage+1}"
        build_union_dataset(
            prev_data_root=current_data_root,
            unlabeled_images_root=unlabeled_images_root,
            preds_root=preds_root,
            keep_names=keep_names,
            out_data_root=next_data_root,
        )

        current_data_root = next_data_root

    print("\n[Iterative] Process Finished.")
    return model_paths


if __name__ == "__main__":
    # Adjust these for your environment
    BASE_DATA_ROOT = "Iterative_dataset\initial_dataset"
    UNLABELED_ROOT = "Iterative_dataset\True_RAW_Data_flattened"
    WORK_ROOT = "Iterative_dataset\iterative_runs"
    EVALUATOR_MODEL = "Iterative_dataset\Balanced_evalutor.pth"

    THRESHOLD = 0.7392

    iterative_unet_training(
        base_data_root=BASE_DATA_ROOT,
        unlabeled_images_root=UNLABELED_ROOT,
        work_root=WORK_ROOT,
        evaluator_model_path=EVALUATOR_MODEL,
        n_stages=5,
        keep_threshold=THRESHOLD,
        device="cuda",
        channels=(32, 64, 128, 256),
        epochs=30,
        lr=1e-4,
        bs=8,
    )
