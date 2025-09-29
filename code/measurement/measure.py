# code/measurement.py

import csv
from pathlib import Path
import cv2
from metro_utils import draw_width_box, draw_gap_box, edge_scan, roughness
from all_structs import EXPECTED_CONTOURS, ROI_RECIPES

PX2NM = 3.90625

def measure_one(img_p: Path, mask_p: Path, struct: str):
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    cnts,_ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) != EXPECTED_CONTOURS[struct]:
        return None

    funcs     = ROI_RECIPES[struct]
    left_box  = funcs['left'] (msk)
    right_box = funcs['right'](msk)
    gap_box   = funcs['gap']  (msk)

    w_px = edge_scan(msk, left_box, right_box)
    g_px = edge_scan(msk, gap_box)

    w_nm = w_px * PX2NM
    g_nm = g_px * PX2NM
    rw   = roughness(w_px)
    rg   = roughness(g_px)

    overlay = draw_width_box(img.copy(), left_box, right_box)
    overlay = draw_gap_box  (overlay, gap_box)
    return w_nm, g_nm, rw, rg, overlay

def measure_all(masks_root: str,
                images_root: str,
                csv_out: str,
                overlays_out: str):
    masks_dir  = Path(masks_root)
    images_dir = Path(images_root)
    out_dir    = Path(overlays_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["wafer","structure","file","width_nm","gap_nm","rough_w","rough_g"])
        for mp in masks_dir.rglob("*.png"):
            rel = mp.relative_to(masks_dir)
            wafer, struct = rel.parts[0], rel.parts[1]
            img_p  = images_dir / rel.with_suffix(".tif")
            result = measure_one(img_p, mp, struct)
            if result is None: continue
            w_nm, g_nm, rw, rg, ov = result
            wr.writerow([wafer,struct,mp.name,f"{w_nm:.3f}",f"{g_nm:.3f}",f"{rw:.3f}",f"{rg:.3f}"])
            ov_p = out_dir / rel
            ov_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ov_p), ov)
