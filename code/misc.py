import cv2
import torch
import torch.nn as nn

import numpy as np

import binarization.img_proc as img_proc
from binarization.img_proc import save_mask_png, load_img

def runU():
    pass


def runB():
    mask = img_proc.binarize_img(
        load_img("C:/Repo/Metrology/data/misc/misc_img.tif"))
    save_mask_png(mask, "C:/Repo/Metrology/data/misc/mask_binarized2.png")



def r():
    from Evaluator import convert_path, Score_stats, print_eval_scores, quality_score
    from pathlib import Path

    labeling_dir = convert_path(r"C:\Repo\Metrology\Images\2.U-Net 6\2.U-Net 6\UNET-6")
    labeling_imgs = sorted(Path(labeling_dir).rglob("*.tif"))
    labeling_imgs = set([ip.name for ip in labeling_imgs])

    double_dir = convert_path(r"C:\Repo\Metrology\Images\3.double U-Net\3.double U-Net\UNET-6-arg")
    double_imgs = sorted(Path(double_dir).rglob("*.tif"))
    double_imgs = set([ip.name for ip in double_imgs])

    print(f"Labeling images: {len(labeling_imgs)}")
    print(f"Double U-Net images: {len(double_imgs)}")

    print("Intersection Set:", len(set(double_imgs) & set(labeling_imgs)))
    print("Difference Set 1:", len(set(double_imgs) - set(labeling_imgs)))
    print("Difference Set 2:", len(set(labeling_imgs) - set(double_imgs)))

    print("double in labeling:", len([ip for ip in double_imgs if ip in labeling_imgs]))
    print("double not in labeling:", len([ip for ip in double_imgs if ip not in labeling_imgs]))

    print("labeling in double:", len([ip for ip in labeling_imgs if ip in double_imgs]))
    print("labeling not in double:", len([ip for ip in labeling_imgs if ip not in double_imgs]))

    print([ip for ip in labeling_imgs if ip not in double_imgs])

def e():
    from Evaluator import convert_path, Score_stats, print_eval_scores, quality_score
    import json

    scores = load_json(convert_path(r"C:\Repo\Metrology\data\JSON\Classical_1203_mask.json"))
    print(max(scores.keys(), key= lambda x: scores[x]), min(scores.keys(), key= lambda x: scores[x]))
    print(Score_stats(scores, title="Classical_1203_mask  Distribution") )

    #TODO CROSS reference with manual curation
    # Evaluator works
    # Train logistical regression on scores

    
    pretty_json = json.load(convert_path(), indent=4)
    print(pretty_json)

def load_json(path):
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return data




if __name__ == "__main__":
    pass





    
