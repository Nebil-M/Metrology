import cv2
import numpy as np
import numpy.typing as npt

from more_itertools import pairwise

import matplotlib.pyplot as plt
import skimage
import scipy
import scipy.signal
import skimage.filters


SCALE = 3.90625


def get_inner_outer(img) -> npt.ArrayLike:
    img = cv2.equalizeHist(img)
    img = cv2.blur(img, (15, 15))
    max_size = 3000

    upper = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, 0
    )
    upper = skimage.morphology.remove_small_objects(upper > 0, max_size) * np.uint8(255)
    upper = skimage.morphology.remove_small_holes(upper > 0, max_size) * np.uint8(255)

    lower = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -10
    )
    lower = skimage.morphology.remove_small_objects(lower > 0, max_size) * np.uint8(255)
    lower = skimage.morphology.remove_small_holes(lower > 0, max_size) * np.uint8(255)

    return lower, upper


def binarize_img_double_thresh(img):
    l, u = get_inner_outer(img)

    n, labeled = cv2.connectedComponents(u)

    out = np.zeros_like(img)
    for i in range(1, n):
        if np.any((labeled == i) & l):
            out[labeled == i] = 255

    return out


def binarize_img(img, block_size=127, c_val=-10, area_thresh=3000) -> npt.ArrayLike:
    # img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 3)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
    )
    img = skimage.morphology.remove_small_objects(img > 0, area_thresh) * np.uint8(255)
    img = skimage.morphology.remove_small_holes(img > 0, area_thresh) * np.uint8(255)
    return img


def binarize_img_adap_hist(img) -> npt.ArrayLike:
    img = cv2.createCLAHE(clipLimit=5, tileGridSize=(10, 10)).apply(img)
    cutoff = 0.9 * cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    img = img > cutoff
    img = img * np.uint8(255)
    img = cv2.erode(img, np.ones((2, 2)))
    img = skimage.morphology.remove_small_objects(img.astype(bool), 2000) * np.uint8(
        255
    )
    img = cv2.dilate(img, np.ones((2, 2)))
    img = skimage.morphology.remove_small_holes(img.astype(bool), 100) * np.uint8(255)
    img = cv2.medianBlur(img, 25)
    return img


def binarize_test(img, grid_size, clip_limit):
    org_img = np.copy(img)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.fastNlMeansDenoising(img, h=20, searchWindowSize=7)
    img = cv2.createCLAHE(clip_limit, (grid_size, grid_size)).apply(img)
    # img = cv2.adaptiveThreshold(
    #     img,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     127,
    #     0,
    # )
    # img = skimage.morphology.remove_small_objects(img.astype(bool), 2500) * np.uint8(
    #     255
    # )
    # img = skimage.morphology.remove_small_holes(img.astype(bool), 2500) * np.uint8(255)
    # img = cv2.medianBlur(img, 3)
    # img = skimage.segmentation.mark_boundaries(org_img, img)
    return img


def get_most_prom_peaks(x, n, min_height=0):
    """Finds the best n peaks in x of a certain minimum height"""
    peaks, data = scipy.signal.find_peaks(
        x, prominence=0, height=min_height, distance=10
    )
    prominences = list(zip(peaks, data["prominences"]))
    prominences.sort(key=lambda x: x[1], reverse=True)
    best_peaks = [x[0] for x in prominences[:n]]
    best_peaks.sort()

    return best_peaks


def find_intersect_col(img, col):
    """Finds all nonzero indices of a certain column of an image"""
    return np.nonzero(img[:, col])[0]


def find_intersect_row(img, row):
    """Finds all nonzero indices of a certain row of an image"""
    return np.nonzero(img[row, :])[0]


def find_vert_width_ROI(
    contour: npt.ArrayLike, start, grow_left, grow_right, max_slope, max_width=100
):
    """Finds the upper and lower boundary points of a binarized image of a contour,
    starting from column `start`"""

    l = start
    r = start
    u_points = []  # list of top contour points
    d_points = []  # list of bottom contour points

    # find intersections along start column and check at least two
    intersections = find_intersect_col(contour, start)
    if len(intersections) < 2:
        return np.array(u_points), np.array(d_points)

    u, d = intersections[[0, -1]]  # get outermost intersections
    u_points.append([start, u])
    d_points.append([start, d])

    # grow region until max width, or change in width too drastic (max_slope)
    while r - l <= max_width:
        if grow_left:
            l -= 1
            u, d = find_intersect_col(contour, l)[[0, -1]]
            u_mean = np.mean(u_points, axis=0)[1]
            d_mean = np.mean(d_points, axis=0)[1]
            if abs(u - u_mean) > max_slope or abs(d - d_mean) > max_slope:
                break

            u_points.append([l, u])
            d_points.append([l, d])

        if grow_right:
            r += 1
            u, d = find_intersect_col(contour, r)[[0, -1]]
            u_mean = np.mean(u_points, axis=0)[1]
            d_mean = np.mean(d_points, axis=0)[1]
            if abs(u - u_mean) > max_slope or abs(d - d_mean) > max_slope:
                break

            u_points.append([r, u])
            d_points.append([r, d])

    return np.array(u_points), np.array(d_points)


def find_hor_width_ROI(
    contour: npt.ArrayLike, start, grow_up, grow_down, max_slope, max_width=100
):
    """Finds the left and right boundary points of a binarized image of a contour,
    starting from row `start`"""

    # do some transformations to make region vertical
    u, d = find_vert_width_ROI(
        contour.T, start, grow_up, grow_down, max_slope, max_width
    )
    return u[:, ::-1], d[:, ::-1]


def find_gap_ROI(l_contour, r_contour, max_diff=10):
    """Finds the left and right boundary points of the gap between two binarized images"""
    rightmost = np.max(np.nonzero(l_contour)[1])

    left_points = []
    right_points = []

    for y in range(512):
        l_intersect = find_intersect_row(l_contour, y)
        r_intersect = find_intersect_row(r_contour, y)
        # make sure intersectiosn are valid
        if len(l_intersect) > 0 and len(r_intersect) > 0:
            # only use rows where the left gap point is near the right edge of
            # the left contour
            if rightmost - l_intersect[-1] < max_diff:
                left_points.append([l_intersect[-1], y])
                right_points.append([r_intersect[0], y])

    return np.array(left_points), np.array(right_points)


def find_parallel_gap_ROI(l_contour, r_contour, start, max_slope, max_width=100):
    """Finds the left and right boundary points for a parallel gap, starting
    from row `start`"""
    u = start
    d = start
    l_points = []  # list of left contour points
    r_points = []  # list of right contour points

    # find intersections along start row and check at least two
    intersect_l = find_intersect_row(l_contour, start)
    intersect_r = find_intersect_row(r_contour, start)
    if len(intersect_l) < 1 or len(intersect_r) < 1:
        return np.array(l_points), np.array(r_points)

    l = intersect_l[-1]
    r = intersect_r[0]
    l_points.append([l, start])
    r_points.append([r, start])

    # grow region until max_width reached or gap changes too quickly (max_slope)
    while d - u <= max_width:
        u -= 1
        l = find_intersect_row(l_contour, u)[-1]
        r = find_intersect_row(r_contour, u)[0]
        l_mean = np.mean(l_points, axis=0)[0]
        r_mean = np.mean(r_points, axis=0)[0]
        if abs(l - l_mean) > max_slope or abs(r - r_mean) > max_slope:
            break

        l_points.append([l, u])
        r_points.append([r, u])

        d += 1
        l = find_intersect_row(l_contour, u)[-1]
        r = find_intersect_row(r_contour, u)[0]
        l_mean = np.mean(l_points, axis=0)[0]
        r_mean = np.mean(r_points, axis=0)[0]
        if abs(l - l_mean) > max_slope or abs(r - r_mean) > max_slope:
            break

        l_points.append([l, d])
        r_points.append([r, d])

    return np.array(l_points), np.array(r_points)


def get_mean_diff(l, r):
    """Finds the mean distance between the left and right points"""
    return SCALE * np.mean(np.linalg.norm(r - l, axis=1))


def get_roughness(l, r, dim):
    return SCALE * np.std(r - l, axis=dim)


def draw_width_ROI(img: npt.NDArray, l, r, value):
    ul = np.min(l, axis=0) - [10, 10]
    dr = np.max(r, axis=0) + [10, 10]
    img = cv2.rectangle(img, ul, dr, (0, 192, 0), 2)

    for i in range(len(l)):
        if i % 9 == 0:
            img = cv2.line(img, l[i], r[i], (0, 128, 255), 2)

    img = cv2.putText(
        img,
        f"{value:.2f}nm",
        ul - [0, 5],
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=(255, 255, 255),
        thickness=2,
    )
    return img


def draw_gap_ROI(img: npt.NDArray, left, right, value):
    ul = np.min(left, axis=0) - [10, 10]
    dr = np.max(right, axis=0) + [10, 10]
    img = cv2.rectangle(img, ul, dr, (0, 192, 0), 2)
    for i, (l, r) in enumerate(zip(left, right)):
        if i % 5 == 0:
            img = cv2.line(img, l, r, (192, 128, 0), 2)

    img = cv2.putText(
        img,
        f"{value:.2f}nm",
        (ul[0], dr[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=(255, 255, 255),
        thickness=2,
    )
    return img
