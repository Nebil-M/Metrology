import time
from itertools import product
from pathlib import Path

import numpy as np
import cv2
from cv2.typing import MatLike
import skimage

MAX_STRUCT_WIDTH = 120


def load_img(path: str) -> MatLike:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def remove_small_objects(img, area_thresh, reverse=False) -> MatLike:
    if reverse:
        img = 255 - img

    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cont in contours:
        if cv2.contourArea(cont) < area_thresh:
            img = cv2.fillPoly(img, [cont], 0)

    if reverse:
        img = 255 - img

    return img


def remove_small_connected(img, area_thresh) -> MatLike:
    count, labeled, stats, centroids = cv2.connectedComponentsWithStats(img)

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < area_thresh:
            labeled[labeled == i] = 0

    return 255 * (labeled > 0).astype(np.uint8)


def binarize_img(img) -> MatLike:
    """Binarizes an image where each pixel is either 0 or 255."""
    max_salt_size = 50
    max_pepper_size = 500

    img = cv2.medianBlur(img, 3)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)[1]
    img = remove_small_objects(img, max_salt_size, reverse=False)
    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    img = remove_small_objects(img, max_pepper_size, reverse=True)

    return img


def depth_fill(img) -> MatLike:
    """Fiils holes, skipping"""

    contours, tree = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def calc_contour_depth(i, tree):
        # no parent contour
        if tree[0, i, 3] == -1:
            return 0

        # else depth is 1 + parent depth
        return calc_contour_depth(tree[0, i, 3], tree) + 1

    depths = [calc_contour_depth(i, tree) for i in range(len(contours))]
    draw_c = []
    for i in range(len(contours)):
        if depths[i] % 4 in [1, 2]:
            draw_c.append(contours[i])

    return cv2.fillPoly(np.zeros_like(img), draw_c, 255)


def keep_long_thin(img, threshold=MAX_STRUCT_WIDTH) -> MatLike:
    """Fix for TCL AND FLG that detects rectangular structures going across entire image."""

    out = np.zeros_like(img)
    count, labeled, stats, centroids = cv2.connectedComponentsWithStats(255 - img)
    for i in range(1, count):
        if (  # vertical bbox
            stats[i, cv2.CC_STAT_HEIGHT] == img.shape[0]
            and stats[i, cv2.CC_STAT_WIDTH] < threshold
        ) or (  # hor bbox
            stats[i, cv2.CC_STAT_WIDTH] == img.shape[1]
            and stats[i, cv2.CC_STAT_HEIGHT] < threshold
        ):
            out[labeled == i] = 255
    return out


def segment_img(img) -> MatLike:
    padded = np.pad(binarize_img(img), 512, mode="symmetric")
    depth = depth_fill(padded)
    thin = keep_long_thin(padded)

    return (depth | thin)[512:1024, 512:1024]


def skeletonize(img, threshold=64, pad_margin=128) -> MatLike:
    "Skeletonizes image and prunes to remove branches smaller than threshold"
    img = np.pad(img, pad_margin, mode="edge")
    img = cv2.medianBlur(img, 5)

    # t0 = time.time()
    img = skimage.morphology.skeletonize(img).astype(np.uint8)
    # print(f"skel: {time.time() - t0:.3f}")

    # prune branches until no changes
    while True:
        new_img = prune(img, threshold)
        if np.all(new_img == img):
            break
        img = new_img

    img = img[pad_margin:-pad_margin, pad_margin:-pad_margin]
    return img.astype(np.uint8) * 255


def prune(skel: np.ndarray[bool], threshold=64) -> np.ndarray[bool]:
    filtered = cv2.filter2D(
        skel,
        ddepth=-1,
        kernel=np.array(
            [
                [1, 1, 1],
                [1, 10, 1],
                [1, 1, 1],
            ]
        ),
    )

    nodes = filtered > 12
    ends = filtered == 11
    branches = skel ^ nodes

    for i, j in np.argwhere(ends):
        _, result, mask, _ = cv2.floodFill(np.copy(branches), None, (j, i), 0, flags=8)
        mask = mask[1:-1, 1:-1]  # mask of current branch
        if np.count_nonzero(mask) < threshold:  # if small branch, then remove it
            branches = result

    return skimage.morphology.skeletonize(branches | nodes).astype(np.uint8)


def count_skeleton_nodes(img):
    filtered = cv2.filter2D(
        img // 255,
        ddepth=-1,
        kernel=np.array(
            [
                [1, 1, 1],
                [1, 10, 1],
                [1, 1, 1],
            ]
        ),
        borderType=cv2.BORDER_CONSTANT,
    )
    nodes = filtered > 12
    return np.count_nonzero(nodes)


def count_regions(img):
    return cv2.connectedComponents(img)[0] - 1


def post_process_img(pred, binarized):
    img = (pred > 0.5).astype(np.uint8) * 255
    img = remove_small_objects(img, 3000, reverse=1)
    filled = remove_small_objects(img | binarized, 10000, reverse=0)
    img = img | filled
    return img | binarized

# save code

def save_mask_png(mask: np.ndarray, out_path: str) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ensure uint8 0/255
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out), mask)
    return str(out)