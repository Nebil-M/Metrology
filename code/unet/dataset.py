from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np, torch

class SEMDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str):
        imgs = sorted(Path(img_dir).rglob("*.tif"))
        lab_map = {p.stem: p for p in Path(mask_dir).rglob("*.tif")}  # or "*.tif" if your masks are tif

        imgs = imgs
        pairs = [(ip, lab_map[ip.stem]) for ip in imgs if ip.stem in lab_map]  # limit to 100 pairs for testing
        assert pairs, "No matched image/mask stems"
        self.imgs, self.masks = zip(*pairs)
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        x = np.array(Image.open(self.imgs[idx]).convert("L"), dtype=np.float32)/255.0
        y = np.array(Image.open(self.masks[idx]).convert("L"), dtype=np.float32)/255.0
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])
