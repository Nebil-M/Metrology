
from unet_training import train_stage
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def run_train():
    train_stage(
        data_root=r"C:/Repo/Metrology/data/First_Unet",   # has /image/*.tif and /label/*.tif
        model_output=r"C:/Repo/Metrology/models/Double_Unet_Nebil11.pth",
        epochs=50, lr=1e-4, bs=8,
        device="cuda", channels=(32,64,128,256)
    )

def run_infer():
    from unet_inference import infer_all
    infer_all(
        model_pth=r"C:/Repo/Metrology/models/unet_tgap_shuhan104.pth",
        images_root=r"C:/Repo/Metrology/data/TGAP_ALL",
        save_root=r"C:/Repo/Metrology/data/Comporation/104_infer",
        device="cuda", channels=(32,64,128,256), thresh=0.5
    )


from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

def _read_gray_robust(path: str) -> np.ndarray:
    arr = None
    try:
        import tifffile as tiff
        arr = tiff.imread(path)
    except Exception:
        pass
    if arr is None:
        try:
            from PIL import Image
            arr = np.array(Image.open(path).convert("F"), dtype=np.float32)
        except Exception:
            pass
    if arr is None:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        try:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        except Exception:
            arr = arr.mean(axis=2)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.0:
            arr = arr / (255.0 if vmax <= 255.0 else vmax)
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr

from typing import List, Tuple, Optional
from pathlib import Path
from math import ceil
import numpy as np

def display_sem_and_masks(
    images: List[Tuple[str, str | Path]],
    masks:  List[Tuple[str, str | Path]],
    title: Optional[str] = None,
    img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    mask_exts=(".png", ".tif", ".tiff"),
    cols: int = 4,                 # pairs per row
    base_h: float = 2.6,           # height (inches) per row
    name_pos: str = "center",
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import cv2

    def _resolve(p: str | Path, exts: tuple[str, ...]) -> Optional[Path]:
        p = Path(p)
        if p.suffix:  # given with extension
            return p if p.exists() else None
        for ext in exts:
            q = p.with_suffix(ext)
            if q.exists():
                return q
        return None

    # --- name→path maps + order ---
    img_map, img_order = {}, []
    for name, p in images:
        if name not in img_map:
            img_map[name] = Path(p); img_order.append(name)

    msk_map, mask_only_order = {}, []
    for name, p in masks:
        if name not in msk_map:
            msk_map[name] = Path(p)
            if name not in img_map:
                mask_only_order.append(name)

    names = img_order + mask_only_order
    if not names:
        print("Nothing to display."); return

    # --- build tiles once to know aspect ---
    tiles = []
    for name in names:
        ip = _resolve(img_map.get(name), img_exts) if name in img_map else None
        mp = _resolve(msk_map.get(name), mask_exts) if name in msk_map else None
        I = _read_gray_robust(str(ip)) if ip is not None else None
        M = _read_gray_robust(str(mp)) if mp is not None else None
        if I is not None and M is not None:
            # match heights for concat
            if I.shape[0] != M.shape[0]:
                h = I.shape[0]
                w = int(round(M.shape[1] * (h / M.shape[0])))
                M = cv2.resize(M, (w, h), interpolation=cv2.INTER_NEAREST)
            tile = np.concatenate([I, M], axis=1)
        else:
            tile = I if I is not None else M
        tiles.append((name, tile))

    # median pair aspect (width/height) to size figure so it fills screen nicely
    aspects = [t.shape[1] / t.shape[0] for _, t in tiles if t is not None]
    pair_aspect = np.median(aspects) if aspects else 1.5

    # --- figure/grid: tiny margins + minimal inter-column gap ---
    n_pairs = len(tiles)
    cols = max(1, int(cols))
    rows = ceil(n_pairs / cols)
    fig_w = cols * pair_aspect * base_h
    fig_h = rows * base_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(rows, cols, figure=fig, wspace=0.02, hspace=0.22)

    for k, (name, tile) in enumerate(tiles):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(outer[r, c])
        if tile is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{name}\n(missing)", ha="center", va="center", fontsize=8)
            continue
        ax.imshow(tile, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        # make the axes box match the tile aspect -> no internal side gaps
        ax.set_box_aspect(tile.shape[0] / tile.shape[1])
        # small name label above tile
        ha = "center" if name_pos == "center" else "left"
        x  = 0.5 if ha == "center" else 0.0
        ax.text(x, 1.02, name, transform=ax.transAxes, ha=ha, va="bottom", fontsize=8)

    # very small outer margins so the content uses the canvas
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96 if title else 0.98,
                        bottom=0.02, wspace=0.02, hspace=0.22)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig
def load_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data

def display_best():
    scores = list(load_json("C:/Repo/Metrology/data/JSON/104_1200.json").keys())[:20]
    
    root_img = Path("C:/Repo/Metrology/data/Classical_1200/image")
    root_mask = Path("C:/Repo/Metrology/data/Classical_1200/104_infer")

    images = [(name, root_img / f"{name}") for name in scores]
    masks = [(name, root_mask / f"{name.rstrip("tif") + "png"}") for name in scores]
    fig = display_sem_and_masks(images, masks, title="Small Unet 104_1200 SEM Images and Masks")
    plt.show()


def run_display():
    from pathlib import Path
    images = [
        ("SEM1", Path("C:/Repo/Metrology/data/Comparation/Classical_infer/E04_M2162-02MS.tif")),
        ("SEM2", Path("C:/Repo/Metrology/data/Comparation/Classical_infer/E04_M2162-02MS.tif")),
    ]
    masks = [
        ("SEM1", Path("C:/Repo/Metrology/data/TGAP_ALL/E04_M2162-02MS.tif")),
        ("SEM2", Path("C:/Repo/Metrology/data/TGAP_ALL/E04_M2162-02MS.tif")),
    ]
    fig = display_sem_and_masks(images, masks, title="SEM Images and Masks")
    plt.show()
if __name__=="__main__":
    display_best()
