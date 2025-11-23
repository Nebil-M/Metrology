from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import csv


class QualityDataset(Dataset):
    """
    Dataset of (raw image, mask, label) triples.

    Expects:
        img_dir / *.tif
        mask_dir / *.tif
        labels.csv with columns: filename,label
    """

    def __init__(self, img_dir: str, mask_dir: str, label_csv: str):
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        label_csv = Path(label_csv)

        imgs = sorted(img_dir.rglob("*.tif"))
        mask_map = {p.stem: p for p in mask_dir.rglob("*.tif")}

        label_map: dict[str, int] = {}
        with open(label_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = Path(row["filename"]).stem
                lab = row["label"].strip().lower()
                if lab not in ("good", "bad"):
                    raise ValueError(f"Unknown label '{lab}' for {stem}")
                label_map[stem] = 1 if lab == "good" else 0

        samples = []
        for ip in imgs:
            stem = ip.stem
            if stem in mask_map and stem in label_map:
                samples.append((ip, mask_map[stem], label_map[stem]))

        if not samples:
            raise RuntimeError("No (image, mask, label) triples found")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        img = np.array(Image.open(img_path).convert("L"),
                       dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"),
                        dtype=np.float32) / 255.0

        img_t = torch.from_numpy(img[None, ...])   # (1,H,W)
        mask_t = torch.from_numpy(mask[None, ...]) # (1,H,W)
        y = torch.tensor(float(label), dtype=torch.float32)

        return img_t, mask_t, y
