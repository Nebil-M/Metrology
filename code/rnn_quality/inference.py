from typing import Literal
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .model import ImageMaskRNNClassifier


def _load_gray(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None, ...])  # (1,H,W)


def predict_pair(
    model_pth: str,
    img_path: str,
    mask_path: str,
    device: Literal["cpu", "cuda"] = "cpu",
    threshold: float = 0.5,
):
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = ImageMaskRNNClassifier().to(device_t).eval()

    state = torch.load(model_pth, map_location=device_t)
    state = state.get("model_state", state.get("state_dict", state))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    img = _load_gray(img_path).unsqueeze(0).to(device_t)
    mask = _load_gray(mask_path).unsqueeze(0).to(device_t)

    with torch.no_grad():
        logits = model(img, mask)
        prob = torch.sigmoid(logits)[0].item()

    label = "good" if prob >= threshold else "bad"
    return prob, label
