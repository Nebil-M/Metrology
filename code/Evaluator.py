from collections import OrderedDict
import numpy as np, cv2, math
from typing import Dict, Tuple

_EPS = 1e-6

def _to_float01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0: x /= 255.0
    return x

def _to_binary01(m: np.ndarray) -> np.ndarray:
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8)
    else:
        m = (m > 127).astype(np.uint8)
    return m

def _boundary_and_rings(m01: np.ndarray, ring_out_iters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    k = np.ones((3,3), np.uint8)
    er = cv2.erode(m01, k, iterations=1)
    di = cv2.dilate(m01, k, iterations=1)
    boundary = ((di - er) > 0)
    ring_out = ((cv2.dilate(m01, k, iterations=ring_out_iters) > 0) & (~m01.astype(bool)))
    return boundary, ring_out

def _grad_mag(i01: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(i01, (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def _connected_stats(m01: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
    # stats: [label, CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA], label 0 is background
    return num, labels, stats

def _largest_component_mask(m01: np.ndarray) -> Tuple[np.ndarray, int]:
    num, labels, stats = _connected_stats(m01)
    if num <= 1:
        return np.zeros_like(m01), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    return (labels == idx).astype(np.uint8), int(areas[idx-1])

def _fill_holes(m01: np.ndarray) -> np.ndarray:
    im = (m01 * 255).astype(np.uint8)
    h, w = im.shape
    tmp = im.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(tmp, mask, (0,0), 255)
    holes = cv2.bitwise_not(tmp)
    filled = cv2.bitwise_or(im, holes)
    return (filled > 0).astype(np.uint8)

def _local_variance(i01: np.ndarray, k: int = 7) -> np.ndarray:
    mean = cv2.blur(i01, (k,k))
    mean2 = cv2.blur(i01*i01, (k,k))
    var = mean2 - mean*mean
    np.maximum(var, 0.0, out=var)
    return var

# 1) Boundary Gradient Ratio (BGR) in [0,1]
def feature_bgr(image: np.ndarray, mask: np.ndarray) -> float:
    I = _to_float01(image); M = _to_binary01(mask)
    boundary, ring_out = _boundary_and_rings(M)
    if not boundary.any():
        return 0.0
    G = _grad_mag(I)
    gb = float(G[boundary].mean())
    gbg = float(G[ring_out].mean()) if ring_out.any() else float(G.mean())
    x = (gb - gbg) / (gb + gbg + _EPS)                      # [-1,1]
    return float(np.clip(0.5*(x + 1.0), 0.0, 1.0))          # [0,1]

# 2) Inside–Outside Contrast Effect size (IOCE) in [0,1]
def feature_ioce(image: np.ndarray, mask: np.ndarray) -> float:
    I = _to_float01(image); M = _to_binary01(mask).astype(bool)
    if not M.any():
        return 0.0
    _, ring_out = _boundary_and_rings(M.astype(np.uint8))
    inside = I[M]; outside = I[ring_out] if ring_out.any() else I[~M]
    mu_in, mu_out = float(inside.mean()), float(outside.mean())
    s_in = float(inside.std() + _EPS); s_out = float(outside.std() + _EPS)
    pooled = math.sqrt((s_in**2 + s_out**2) / 2.0)
    d = abs(mu_in - mu_out) / (pooled + _EPS)               # Cohen's d
    return float(d / (d + 1.0))                             # [0,1]

# 3) Small-Component Area Ratio (SCAR) → good score in [0,1]
def feature_scar(image: np.ndarray, mask: np.ndarray, k_frac: float = 0.001, k_abs: int = 2000) -> float:
    M = _to_binary01(mask)
    total = int(M.sum())
    if total == 0:
        return 0.0
    H, W = M.shape
    thr = max(int(k_frac*H*W), k_abs)
    num, labels, stats = _connected_stats(M)
    small_area = int(stats[1:, cv2.CC_STAT_AREA][stats[1:, cv2.CC_STAT_AREA] < thr].sum()) if num > 1 else 0
    ratio = small_area / (total + _EPS)                     # 0..1 (higher worse)
    good = 1.0 - min(1.0, ratio/0.2)                        # penalize if >20% tiny specks
    return float(np.clip(good, 0.0, 1.0))

# 4) Largest-Component Dominance (LCD) in [0,1]
def feature_lcd(image: np.ndarray, mask: np.ndarray) -> float:
    M = _to_binary01(mask)
    total = int(M.sum())
    if total == 0:
        return 0.0
    _, labels, stats = _connected_stats(M)
    if len(stats) <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(largest / (total + _EPS))

# 5) Hole Fraction (HF) → good score in [0,1]
def feature_hole_fraction(image: np.ndarray, mask: np.ndarray) -> float:
    M = _to_binary01(mask)
    if M.sum() == 0:
        return 0.0
    filled = _fill_holes(M)
    hf = (int(filled.sum()) - int(M.sum())) / (int(filled.sum()) + _EPS)   # 0..1 (higher = more holes)
    return float(np.clip(1.0 - hf, 0.0, 1.0))

# 6) Boundary Roughness / Compactness (BR) in [0,1]
def feature_compactness(image: np.ndarray, mask: np.ndarray) -> float:
    M = _to_binary01(mask)
    main, area = _largest_component_mask(M)
    if area == 0:
        return 0.0
    cnts, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    P = float(cv2.arcLength(cnts[0], True))
    C = float(4.0 * math.pi * area / (P*P + _EPS))          # 0..1 (circle=1)
    return float(np.clip(C, 0.0, 1.0))

# 7) Intensity Texture Separation (ITS) in [0,1]
def feature_texture_sep(image: np.ndarray, mask: np.ndarray, k: int = 7) -> float:
    I = _to_float01(image); M = _to_binary01(mask).astype(bool)
    if not M.any():
        return 0.0
    V = _local_variance(I, k=k)
    _, ring_out = _boundary_and_rings(M.astype(np.uint8))
    inside = V[M]; outside = V[ring_out] if ring_out.any() else V[~M]
    diff = abs(float(inside.mean()) - float(outside.mean()))
    return float(diff / (diff + 1.0))                       # [0,1]

# 8) Threshold-free Area Sanity (TAS) in [0,1]
def feature_tas(image: np.ndarray, mask: np.ndarray, mu: float = 0.10, sigma: float = 0.10) -> float:
    M = _to_binary01(mask)
    H, W = M.shape
    af = float(M.sum()) / float(H*W + _EPS)                 # 0..1
    z = (af - mu) / (sigma + _EPS)
    score = math.exp(-0.5 * z*z)                            # bell around mu
    return float(np.clip(score, 0.0, 1.0))

def quality_score(image: np.ndarray, mask: np.ndarray,
                  weights: Dict[str, float] = None) -> Dict[str, float]:
    feats = {
        "bgr":  feature_bgr(image, mask),
        "ioce": feature_ioce(image, mask),
        "scar": feature_scar(image, mask),
        "lcd":  feature_lcd(image, mask),
        "br":   feature_compactness(image, mask),
        "hf":   feature_hole_fraction(image, mask),
        "its":  feature_texture_sep(image, mask),
        "tas":  feature_tas(image, mask),
    }
    default_w = {"bgr":0.25, "ioce":0.20, "scar":0.15, "lcd":0.15, "br":0.10, "hf":0.10, "its":0.05, "tas":0.05}
    w = (weights or default_w).copy()
    s = sum(w.get(k,0.0) for k in feats.keys());  w = {k: (w.get(k,0.0)/s if s>0 else 0.0) for k in feats.keys()}
    score = float(sum(w[k]*feats[k] for k in feats.keys()))
    feats["score"] = score
    return feats

def read_gray_robust(path: str) -> np.ndarray:
    arr = None
    # 1) try tifffile (best for 16/32-bit TIFFs)
    try:
        import tifffile as tiff
        arr = tiff.imread(path)
    except Exception:
        pass
    # 2) fallback: Pillow
    if arr is None:
        try:
            from PIL import Image
            arr = np.array(Image.open(path).convert("F"), dtype=np.float32)  # "F" = 32-bit float gray
        except Exception:
            pass
    # 3) fallback: OpenCV (works for 8/16-bit, some TIFFs)
    if arr is None:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read image: {path}")

    # to single-channel
    if arr.ndim == 3:
        # if Pillow gave RGBA float, average channels; if OpenCV BGR, convert to gray
        try:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        except Exception:
            arr = arr.mean(axis=2)

    # normalize to [0,1]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:  # float types
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.0:
            arr = arr / (255.0 if vmax <= 255.0 else vmax)
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    return arr

def printer(img_path, mask_path):
    img = read_gray_robust(img_path)
    mask = read_gray_robust(mask_path)
    res = quality_score(img, mask)
    print(res["score"], str(mask_path).split("\\")[-1])  
    
def print_eval_scores(
    image_dir: str,
    mask_dir: str,
    scorer=None,                       # pass quality_score here if it's not in scope
    img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    mask_exts=(".png", ".tif", ".tiff"),
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    show_subscores: bool = False,
    print_scores: bool = True
):
    if scorer is None:
        # if quality_score is in the same file, it will already be in scope
        try:
            from Evaluator import quality_score as scorer  # adjust module name if needed
        except Exception:
            raise ImportError("Provide scorer=quality_score or import it before calling print_eval_scores().")

    imgs = []
    for ext in img_exts:
        imgs += list(Path(image_dir).rglob(f"*{ext}"))
    imgs = sorted(imgs)

    mask_by_stem = {}
    for ext in mask_exts:
        for p in Path(mask_dir).rglob(f"*{ext}"):
            mask_by_stem[p.stem] = p

    pairs = [(ip, mask_by_stem[ip.stem.lstrip("mask_")]) for ip in imgs if ip.stem.lstrip("mask_") in mask_by_stem]
    if not pairs:
        print("No matched image/mask stems found."); return
    if shuffle:
        random.Random(seed).shuffle(pairs)
    if limit:
        pairs = pairs[:limit]

    from collections import OrderedDict
    scores = OrderedDict()
    for ip, mp in pairs:
        try:
            I = read_gray_robust(str(ip))
            M = read_gray_robust(str(mp))
            res = scorer(I, M)  # expects dict with 'score' and subscores
            scores[ip.name] = res['score']

            if show_subscores:
                print(f"{ip.name}\tscore={res['score']:.3f}\t"
                      f"bgr={res['bgr']:.3f}\tioce={res['ioce']:.3f}\t"
                      f"scar={res['scar']:.3f}\tlcd={res['lcd']:.3f}\t"
                      f"br={res['br']:.3f}\thf={res['hf']:.3f}\t"
                      f"its={res['its']:.3f}\ttas={res['tas']:.3f}")
            elif print_scores:
                print(f"{ip.name}\t{res['score']:.3f}")
        except Exception as e:
            print(f"{ip.name}\tERROR: {e}")
    scores = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return scores

def r():
    scores = print_eval_scores(
        image_dir=convert_path(r"C:/Repo/Metrology/data/Classical_1200/image"),
        mask_dir=convert_path(r"C:/Repo/Metrology/data/Classical_1200/masks"),
        scorer=quality_score,
        img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        mask_exts=(".png", ".tif", ".tiff"),
        print_scores=False,
    )

    #TODO CROSS reference with manual curation
    # Evaluator works
    # Train logistical regression on scores

    import json
    pretty_json = json.dumps(scores, indent=4)
    print(pretty_json)

    keep = [name for name in scores if scores[name] >= 0.6 ]
    noKeep = [name for name in scores if scores[name] < 0.6 ]

    dir = convert_path(r"C:\Repo\Metrology\Images\2.U-Net 6\2.U-Net 6\UNET-6\image")
    imgs = sorted(Path(dir).rglob("*.tif"))

    curated = [ip.name for ip in imgs]

    print("keep: ", len(keep))
    print("Curated: ", len(curated))
    print("noKeep: ", len(noKeep))
    print("Keep in Curated: ", len([k for k in keep if k in curated]))
    print("Not Keep in Curated: ", len([k for k in noKeep if k in curated]))
    print("Curated in Keep: ", len([k for k in curated if k in keep]))
    print("Curated in NoKeep: ", len([k for k in curated if k in noKeep]))

    print("all in curated", len([k for k in curated if k in keep or k in noKeep]))
    
    print("not curated in noKeep", len([k for k in noKeep if k not in curated]))
    print("not curated in keep", len([k for k in keep if k not in curated]))

def convert_path(path: str) -> str:
    return path.replace("\\", "/") if "\\" in path else path

def Score_stats(scores: Dict[str, float], title = "Score Distribution") -> Tuple[float, float]:
    values = np.array(list(scores.values()))
    mean_score = values.mean()
    std_dev = values.std()
    import matplotlib.pyplot as plt
    plt.hist(values, bins=100, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.text(0.5, 0.95, f"Mean: {mean_score:.3f}, Std Dev: {std_dev:.3f}", ha='center', va='top', transform=plt.gca().transAxes)
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean_score + std_dev, color='green', linestyle='dashed', linewidth=1)
    plt.axvline(mean_score - std_dev, color='green', linestyle='dashed', linewidth=1)
    plt.show()

    return mean_score, std_dev

def savef():
    scores = print_eval_scores(
        image_dir=convert_path(r"C:/Repo/Metrology/data/TGAP_ALL"),
        mask_dir=convert_path(r"C:/Repo/Metrology/data/Comparation/104_infer"),
        scorer=quality_score,
        img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        mask_exts=(".png", ".tif", ".tiff"),
        print_scores=False,
    )
    print(max(scores.keys(), key= lambda x: scores[x]), min(scores.keys(), key= lambda x: scores[x]))
    print(Score_stats(scores, title="104's TGAP Distribution") )

    #TODO CROSS reference with manual curation
    # Evaluator works
    # Train logistical regression on scores

    import json
    pretty_json = json.dumps(scores, indent=4)
    print(pretty_json)
    with open("C:/Repo/Metrology/data/JSON/104_TGAP.json", "w") as f:
        json.dump(scores, f, indent=4)
if __name__ == "__main__":
    from pathlib import Path
    import cv2, numpy as np
    img = Path(convert_path(r"C:\Repo\Metrology\data\misc\514_SEM.tif"))
    mask = Path(convert_path(r"C:\Repo\Metrology\data\misc\singleton\S04_M0960-02MS.png"))
    printer(img, mask)


   
    

    