""" Full assembly of the parts to form the complete network """

"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F
from pathlib import Path

from unet_parts import *
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch, cv2, numpy as np


class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def run_inference(model_pth: str, img_path: str, out_path: str, thresh: float = 0.5, device: str = "cuda"):
    device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = UNet(n_channels=1, n_classes=1).to(device).eval()
    state = torch.load(model_pth, map_location=device)
    state = state.get("model_state", state.get("state_dict", state))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    mask = (prob > thresh).astype(np.uint8) * 255
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    print(f"Saved: {out_path}")


def infer_all(model_pth: str, images_root: str, out_root: str, thresh: float = 0.5, device: str = "cuda"):
    device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = UNet(n_channels=1, n_classes=1).to(device).eval()

    state = torch.load(model_pth, map_location=device)
    state = state.get("model_state", state.get("state_dict", state))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    images_root = Path(images_root)
    out_root = Path(out_root)
    for tif in images_root.rglob("*.tif"):
        img = cv2.imread(str(tif), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"SKIP (unreadable): {tif}")
            continue
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        mask = (prob > thresh).astype(np.uint8) * 255
        out_path = out_root / tif.relative_to(images_root)
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), mask)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    infer_all(
        model_pth=r"C:/Repo/Metrology/models/unet_tgap_b_red_contrast.pth",
        images_root=r"C:/Repo/Metrology/data/TGAP_ALL",
        out_root=r"C:/Repo/Metrology/data/Comporation/104_infer",
        device="cuda", thresh=0.5
    )

