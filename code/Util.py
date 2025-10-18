import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import binarization.img_proc as img_proc
from pathlib import Path
import glob
import os

from unet.unet_training import train_stage
from unet.unet_inference import infer_all
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class npyImageManager:
    def __init__(self):
        pass
    
    @staticmethod
    def display_npy_image(path):
        arr = np.load(path, allow_pickle=False)
        img = arr
        if img.ndim == 3:
            img = img[img.shape[0]//2] if arr.shape[0] < 10 else arr[...,0]
        elif img.ndim == 4:
            img = img[0, img.shape[1]//2]
        x = img.astype(np.float32)
        
        mn,mx = float(x.min()), float(x.max())
        x = (x - mn)/(mx - mn + 1e-6)
        x = np.clip(x,0,1)

        mn,mx = float(img.min()), float(img.max())
        plt.imshow(x, cmap="gray")
        plt.title(f"{arr.shape} {arr.dtype} [{mn:.3g},{mx:.3g}] mode=auto")
        plt.axis("off")
        plt.show()


    @staticmethod
    def Load_npy_image(path):
        arr = np.load(path, allow_pickle=False)
        img = arr
        return img


class BinarizationManager:
    def __init__(self):
       pass 

    @staticmethod
    def binarize_image(img):
        mask = img_proc.binarize_img(img)
        return mask
    
    @staticmethod
    def batch_binarize(folder_path: str, out_folder: str):
        paths = glob.glob(f"{folder_path}/*.tif")
        for path in paths:
            img = img_proc.load_img(path)
            mask = img_proc.binarize_img(img)
            img_proc.save_mask_png(mask, os.path.join(out_folder, f"{os.path.basename(path)}"))

class UnetManager:
    def __init__(self):
        pass

    @staticmethod
    def run_train(data_root, model_output):
        """ Run U-Net training with specified data root and model output path. 
        Model output path is complete path of model file including .pth suffix."""
        train_stage(
            data_root = data_root,   # has /image/*.tif and /label/*.tif
            model_output = model_output,
            epochs=50, lr=1e-4, bs=8,
            device="cuda", channels=(32,64,128,256)
        )

    @staticmethod
    def run_infer(model_pth, images_folder_root, save_folder_root,
                  thresh=0.5, channels=(32,64,128,256), device="cuda"):
        """ Run U-Net inference with specified model path and data paths inside the function."""
        infer_all(
            model_pth=model_pth,
            images_root=images_folder_root,
            save_root=save_folder_root,
            device=device, channels=channels, thresh=thresh
        )


class evalutorManager:
    def __init__(self):
        pass