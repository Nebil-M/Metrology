# code/unet/iterative_training.py

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

    Pairs:
      - image_root/**/<name>.(tif/png/...)
      - mask_root/**/<stem>.(tif/png/...)

    Returns a 4-channel tensor [R,G,B,mask] and the image filename (basename).
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
    """
    Same structure as your evaluator:
    ResNet18 with 4-channel conv1 and a single-logit head.
    """
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
    threshold: float = 0.9, # Change this to modify threshold based on how well it does over validation  (a sweep over validation) during training.
) -> Dict[str, int]:
    """
    Use the evaluator model to label each (image, mask) pair as GOOD (1) or BAD (0).

    - Evaluator outputs logits.
    - We compute prob_good = sigmoid(logit).
    - We immediately convert to a hard Good/Bad decision:
          is_good = int(prob_good >= threshold)

    Returns:
        labels: dict mapping image filename -> 1 (Good) or 0 (Bad)

    This is now *binary*; the rest of the pipeline only sees good/bad.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    dataset = EvaluatorInferenceDataset(image_dir, mask_dir)
    if len(dataset) == 0:
        print("[Evaluator] No samples to classify. Returning empty dict.")
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load evaluator weights
    if not os.path.exists(evaluator_model_path):
        raise FileNotFoundError(f"[Evaluator] Model not found at {evaluator_model_path}")

    model = get_evaluator_model().to(device_obj)
    state = torch.load(evaluator_model_path, map_location=device_obj)
    model.load_state_dict(state)
    model.eval()

    labels: Dict[str, int] = {}
    raw_probs: Dict[str, float] = {}  # optional: for logging / inspection

    with torch.no_grad():
        for inputs, fnames in loader:
            inputs = inputs.to(device_obj)
            logits = model(inputs).squeeze(1)  # (B,)

            probs = torch.sigmoid(logits).cpu().numpy()  # P(Good)

            for fname, p in zip(fnames, probs):
                is_good = int(p >= threshold)   # HARD Good/Bad decision
                labels[fname] = is_good
                raw_probs[fname] = float(p)

    print("[Evaluator] Finished classifying masks as Good/Bad.")

    # If you don't want any probabilities persisted, remove raw_probs usage below.
    return labels, raw_probs


# ---------- UNION DATASET BUILDER (unchanged) ----------

def build_union_dataset(
    prev_data_root: Path,
    unlabeled_images_root: Path,
    preds_root: Path,
    keep_names: List[str],
    out_data_root: Path,
) -> None:
    ...
    # (same as your current build_union_dataset; omitted here to save space)
    ...


# ---------- ITERATIVE TRAINING LOOP (now uses Good/Bad directly) ----------

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
    """
    Iterative U-Net training using the evaluator as a Good/Bad classifier.

    IMPORTANT: Good/Bad is decided *inside* classify_good_bad:

        prob_good = sigmoid(logit)
        is_good = (prob_good >= keep_threshold)

    The iterative loop only ever sees binary labels (1=Good, 0=Bad).
    """

    base_data_root = Path(convert_path(base_data_root)).resolve()
    unlabeled_images_root = Path(convert_path(unlabeled_images_root)).resolve()
    work_root = Path(convert_path(work_root)).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"[Iterative] Base dataset:   {base_data_root}")
    print(f"[Iterative] Unlabeled root: {unlabeled_images_root}")
    print(f"[Iterative] Work root:      {work_root}")
    print(f"[Iterative] Stages: {n_stages}, keep_threshold={keep_threshold}")

    assert (base_data_root / "image").exists(), f"{base_data_root}/image not found"
    assert (base_data_root / "label").exists(), f"{base_data_root}/label not found"

    current_data_root = base_data_root
    model_paths: List[Path] = []

    for stage in range(1, n_stages + 1):
        print("\n" + "=" * 80)
        print(f"[Iterative] STAGE {stage}/{n_stages}")
        print("=" * 80)

        # 1) Train U-Net at this stage
        model_pth = work_root / f"unet_stage{stage}.pth"
        print(f"[Iterative] Training stage-{stage} model at {model_pth}")

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
            print(f"[Iterative] Reached final stage ({stage}); stopping.")
            break

        # 2) Run inference on unlabeled images
        preds_root = work_root / f"preds_stage{stage}"
        print(f"[Iterative] Inferring pseudo-labels into {preds_root}")
        infer_all(
            model_pth=str(model_pth),
            images_root=str(unlabeled_images_root),
            save_root=str(preds_root),
            device=device,
            channels=channels,
            thresh=0.5,
        )

        # 3) Classify predicted masks as Good/Bad using the evaluator
        print(f"[Iterative] Classifying predicted masks as Good/Bad...")
        good_bad_labels, raw_probs = classify_good_bad(
            image_dir=str(unlabeled_images_root),
            mask_dir=str(preds_root),
            evaluator_model_path=evaluator_model_path,
            device=device,
            batch_size=16,
            threshold=keep_threshold,   # Good/Bad decided *inside* here
        )

        if not good_bad_labels:
            print("[Iterative] WARNING: no labels returned; "
                  "next stage will reuse previous dataset.")
            keep_names: List[str] = []
        else:
            # (Optional) Save raw probabilities and Good/Bad flags for inspection
            debug_json = work_root / f"stage{stage}_evaluator_output.json"
            with open(debug_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prob_good": raw_probs,
                        "good_bad": good_bad_labels,
                        "threshold": keep_threshold,
                    },
                    f,
                    indent=2,
                )
            print(f"[Iterative] Saved evaluator debug JSON to {debug_json}")

            # 4) ONLY KEEP "GOOD" samples (value == 1)
            keep_names = [name for name, label in good_bad_labels.items() if label == 1]
            print(f"[Iterative] Kept {len(keep_names)} / {len(good_bad_labels)} pseudo-labels "
                  f"(classified as GOOD)")

        # 5) Build new dataset = union(previous_dataset + GOOD pseudo-labels)
        next_data_root = work_root / f"data_stage{stage+1}"
        build_union_dataset(
            prev_data_root=current_data_root,
            unlabeled_images_root=unlabeled_images_root,
            preds_root=preds_root,
            keep_names=keep_names,
            out_data_root=next_data_root,
        )

        current_data_root = next_data_root

    print("\n[Iterative] Done.")
    print("Trained models:")
    for i, p in enumerate(model_paths, start=1):
        print(f"  Stage {i}: {p}")
    return model_paths


if __name__ == "__main__":
    # Adjust these for your environment
    BASE_DATA_ROOT = "/Users/alexstrugacz/ml-research/Metrology/data/UNET-6-arg"
    UNLABELED_ROOT = "/Users/alexstrugacz/ml-research/Metrology/data/True_RAW_Data"
    WORK_ROOT = "/Users/alexstrugacz/ml-research/Metrology/iterative_runs"
    EVALUATOR_MODEL = "/Users/alexstrugacz/ml-research/Metrology/best_evaluator.pth"

    iterative_unet_training(
        base_data_root=BASE_DATA_ROOT,
        unlabeled_images_root=UNLABELED_ROOT,
        work_root=WORK_ROOT,
        evaluator_model_path=EVALUATOR_MODEL,
        n_stages=3,
        keep_threshold=0.6,   # used INSIDE evaluator to decide Good/Bad
        device="cpu",
        channels=(32, 64, 128, 256),
        epochs=50,
        lr=1e-4,
        bs=8,
    )
