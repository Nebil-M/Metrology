import glob
import os

import numpy as np
import matplotlib.pyplot as plt

img_sets = [
    "TGAP_FINAL",
    "TGAP_B",
    "DTGAP_A",
    "DTGAP_B",
    "TTGAP_A",
    "TTGAP_B",
    "FLG_A",
    "FLG_B",
    "TCL_A",
    "TCL_B",
    "UL_A",
    "UL_B",
    "ETE_A",
    "ETE_B",
    "TETE_A",
    "TETE_B",
]

img_set_to_path = {}


def _load_dir():
    img_dir = "/home/erl75/cleanroom_data/raw_data/post_litho_raw/"
    for struct_type in next(os.walk(img_dir))[1]:
        for img_set in next(os.walk(f"{img_dir}{struct_type}"))[1]:
            short_name = next(
                filter(
                    lambda name: name in img_set and name.split("_")[0] == struct_type,
                    img_sets,
                )
            )
            img_set_to_path[short_name] = f"{img_dir}{struct_type}/{img_set}"

    # fix TGAP_FINAL
    img_set_to_path["TGAP_A"] = img_set_to_path["TGAP_FINAL"]


_load_dir()


def imgs_in_set(img_set):
    folder = img_set_to_path[img_set]
    return glob.glob(f"{folder}/*.tif")


def view_imgs(imgs, names=None, sep=True, width=5):
    if not sep:
        plt.figure(figsize=(3 * width, 3 * len(imgs) / width), facecolor=(0, 0, 0))

    for i, img in enumerate(imgs):
        if sep:
            if i % width == 0:
                plt.show()
                plt.figure(figsize=(3 * width, 3))

            plt.subplot(1, width, i % width + 1)

        else:
            plt.subplot((len(imgs) + width - 1) // width, width, i + 1)

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        if names:
            plt.title(names[i].split("/")[-1], fontdict={"fontsize": 8})


def create_rgb_img(r, g, b):
    return np.stack([r, g, b], axis=-1)
