from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class QualityDataset(Dataset):
    """
    Dataset of (raw image, mask, label) triples.
    
    New Structure Expectation:
        root_dir/
            1_Good/
                images/ (contains images)
                masks/  (contains corresponding masks)
            0_Bad/
                images/
                masks/
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.samples = []

        # Define the structure we created: Folder Name -> Label Value
        # 1_Good -> Label 1.0
        # 0_Bad  -> Label 0.0
        class_map = {
            "1_Good": 1.0,
            "0_Bad": 0.0
        }

        print(f"Loading dataset from: {self.root}")

        for class_name, label_value in class_map.items():
            class_path = self.root / class_name
            img_dir = class_path / "images"
            mask_dir = class_path / "masks"

            if not img_dir.exists() or not mask_dir.exists():
                print(f"[WARN] Missing directories for class {class_name}. Skipping.")
                continue

            # 1. Gather all masks first (Map Stem -> Path)
            # We accept any common image extension
            mask_map = {}
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                for p in mask_dir.rglob(ext):
                    mask_map[p.stem] = p

            # 2. Iterate through images and find matching mask
            img_count = 0
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                for img_path in img_dir.rglob(ext):
                    stem = img_path.stem
                    
                    if stem in mask_map:
                        mask_path = mask_map[stem]
                        self.samples.append((img_path, mask_path, label_value))
                        img_count += 1
            
            print(f"   -> Found {img_count} pairs in {class_name}")

        if not self.samples:
            raise RuntimeError(f"No valid (image, mask) pairs found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        # Convert to Grayscale ('L') and normalize 0-1
        # Using PIL ensures we can handle tif/png/jpg identically
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        img_t = torch.from_numpy(img[None, ...])   # Add channel dim: (1,H,W)
        mask_t = torch.from_numpy(mask[None, ...]) # Add channel dim: (1,H,W)
        y = torch.tensor(float(label), dtype=torch.float32)

        return img_t, mask_t, y