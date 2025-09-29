"""
Specific functions for processing each type of structure. Each function generally
works by splitting up the binarized input image into several masked images, each
containing one structure and measuring the widths and gaps based on those masked
images.
"""

from metro_utils import *

import numpy as np
import numpy.typing as npt

DARK_THRESH = 10


def tgap(bin_img: npt.ArrayLike):
    if (
        len(cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        == 1
    ):
        return np.array([]), -1, np.array([]), -1

    mean_cols = np.mean(bin_img, axis=0)
    best_peaks_vert = get_most_prom_peaks(mean_cols, 2, 64)

    r_gap = best_peaks_vert[0]
    for i in range(best_peaks_vert[0], -1, -1):
        if mean_cols[i] < DARK_THRESH:
            r_gap = i
            break

    l_gap = r_gap
    for i in range(r_gap, -1, -1):
        if mean_cols[i] > DARK_THRESH:
            l_gap = i
            break

    # create masked portions
    cutoff = int(np.mean(np.array([l_gap, r_gap])))
    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :cutoff] = bin_img[:, :cutoff]
    r_part[:, cutoff:] = bin_img[:, cutoff:]

    w_result = find_hor_width_ROI(r_part, 150, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_gap_ROI(l_part, r_part, max_diff=5)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def dtgap(bin_img: npt.ArrayLike):
    # get average brightness of columns
    mean_cols = np.mean(bin_img, axis=0)

    # find gap by looking for when the average first goes below, then above threshold
    l_gap = np.argwhere(mean_cols < DARK_THRESH)[0, 0]
    r_gap = l_gap + np.argwhere(mean_cols[l_gap:] > DARK_THRESH)[0, 0]

    cutoff = int(np.mean([l_gap, r_gap]))
    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :cutoff] = bin_img[:, :cutoff]
    r_part[:, cutoff:] = bin_img[:, cutoff:]

    w_result = find_hor_width_ROI(r_part, 150, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_gap_ROI(l_part, r_part)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def ttgap(bin_img: npt.ArrayLike):
    # # get average brightness along rows
    # mean_rows = np.mean(bin_img, axis=1)
    # # find left structures' bounds
    # best_peaks_hor = get_most_prom_peaks(mean_rows, 6, 64)

    # # get 2 peaks closest to the middle and sort
    # middle = sorted(best_peaks_hor, key=lambda x: abs(x - 256))[:2]
    # middle.sort()

    # # get avg brightness of cols within bounds
    # mean_window = np.mean(bin_img[middle[0] : middle[1]], axis=0)
    # # find the edge of left strucutre and bounds of right structure
    # peaks_window = get_most_prom_peaks(mean_window, 3, 64)

    mean_cols = np.mean(bin_img, axis=0)
    l_gap = np.argwhere(mean_cols < DARK_THRESH)[0, 0]
    r_gap = l_gap + np.argwhere(mean_cols[l_gap:] > DARK_THRESH)[0, 0]

    cutoff = int(np.mean([l_gap, r_gap]))

    # create masked portions
    # cutoff = int(np.mean(peaks_window[:2]))
    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :cutoff] = bin_img[:, :cutoff]
    r_part[:, cutoff:] = bin_img[:, cutoff:]

    w_result = find_hor_width_ROI(r_part, 150, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_gap_ROI(l_part, r_part)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def _ntgap(bin_img: npt.ArrayLike):
    """Old code to process TGAP, DTGAP, and TTGAP together"""
    # get average brightness
    mean_cols = np.mean(bin_img, axis=0)

    # find peaks that correspond to right side
    best_peaks = get_most_prom_peaks(mean_cols, 2, 64)

    # find adjacent dark region
    r_gap = best_peaks[0]
    for i in range(best_peaks[0], -1, -1):
        if mean_cols[i] < DARK_THRESH:
            r_gap = i
            break

    l_gap = r_gap
    for i in range(r_gap, -1, -1):
        if mean_cols[i] > DARK_THRESH:
            l_gap = i
            break

    cutoff = int(np.mean([l_gap, r_gap]))

    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :cutoff] = bin_img[:, :cutoff]
    r_part[:, cutoff:] = bin_img[:, cutoff:]

    w_result = find_hor_width_ROI(r_part, 150, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_gap_ROI(l_part, r_part)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def flg(bin_img: npt.ArrayLike):
    # get average brightness across top of image
    mean_cols_top = np.mean(bin_img[:128, :], axis=0)
    # find bounds of middle structure
    best_peaks = get_most_prom_peaks(mean_cols_top, 2)

    # find adjacent dark region
    mean_cols = np.mean(bin_img, axis=0)
    l_gap = best_peaks[1]
    for i in range(best_peaks[1], 512, 1):
        if mean_cols[i] < DARK_THRESH:
            l_gap = i
            break

    r_gap = l_gap
    for i in range(l_gap, 512, 1):
        if mean_cols[i] > DARK_THRESH:
            r_gap = i
            break

    # split image in middle of gap and create masks
    cutoff = int(np.mean([l_gap, r_gap]))

    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :cutoff] = bin_img[:, :cutoff]
    r_part[:, cutoff:] = bin_img[:, cutoff:]

    w_result = find_hor_width_ROI(l_part, 100, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_parallel_gap_ROI(l_part, r_part, 400, 5)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def tcl(bin_img: npt.ArrayLike):
    # get average brightness across top of image
    mean_cols_top = np.mean(bin_img[:128, :], axis=0)

    best_peaks = get_most_prom_peaks(mean_cols_top, 5, 0)

    # gaps are between 2nd and 3rd peaks and 4th and 5th peaks
    l_cut = int(np.mean(best_peaks[1:3]))
    r_cut = int(np.mean(best_peaks[3:5]))

    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[:, :l_cut] = bin_img[:, :l_cut]
    r_part[:, l_cut:r_cut] = bin_img[:, l_cut:r_cut]

    w_result = find_hor_width_ROI(r_part, 100, True, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_parallel_gap_ROI(l_part, r_part, 100, 5)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap


def ul(bin_img: npt.ArrayLike):
    # get mean brightness along columns
    mean_cols = np.mean(bin_img, axis=0)
    # find the boundaries of the central structure and 2 sides of the u
    best_vert_peaks = get_most_prom_peaks(mean_cols, 6, 64)

    l_cut = int(np.mean(best_vert_peaks[1:3]))
    r_cut = int(np.mean(best_vert_peaks[3:5]))

    mean_rows = np.mean(bin_img[128:, l_cut:r_cut], axis=1)
    best_hor_peaks = get_most_prom_peaks(mean_rows, 2)

    # find dark region
    d_gap = best_hor_peaks[0]
    for i in range(best_hor_peaks[0], -1, -1):
        if mean_rows[i] < DARK_THRESH:
            d_gap = i
            break

    u_gap = d_gap
    for i in range(d_gap, -1, -1):
        if mean_rows[i] > DARK_THRESH:
            u_gap = i
            break

    h_cut = int(np.mean([u_gap, d_gap])) + 128

    m_part = np.zeros_like(bin_img)
    d_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part = np.zeros_like(bin_img)
    m_part[:h_cut, l_cut:r_cut] = bin_img[:h_cut, l_cut:r_cut]
    d_part[h_cut:, l_cut:r_cut] = bin_img[h_cut:, l_cut:r_cut]
    r_part[:, r_cut:] = bin_img[:, r_cut:]
    l_part[:, :l_cut] = bin_img[:, :l_cut]

    w_result = find_hor_width_ROI(r_part, 250, True, True, 5)
    width = get_mean_diff(*w_result)
    # function only measures horizontal gaps, so need to transform image and results
    g_result = find_gap_ROI(m_part.T, d_part.T)
    g_result = g_result[0][:, ::-1], g_result[1][:, ::-1]
    gap = get_mean_diff(*g_result)

    # fix for missing right side
    if width < 100:
        w_result = find_hor_width_ROI(l_part, 250, True, True, 5)
        width = get_mean_diff(*w_result)

    return w_result, width, g_result, gap


def ete(bin_img: npt.ArrayLike):
    # find average brightness of columns along middle strip of image
    mean_cols = np.mean(bin_img[128:384], axis=0)

    l_gap = np.argwhere(mean_cols < DARK_THRESH)[0, 0]
    r_gap = l_gap + np.argwhere(mean_cols[l_gap:] > DARK_THRESH)[0, 0]

    # cut = int(np.mean(best_peaks))
    cut = int(np.mean([l_gap, r_gap]))

    l_part = np.zeros_like(bin_img)
    r_part = np.zeros_like(bin_img)
    l_part[128:384, :cut] = bin_img[128:384, :cut]
    r_part[128:384, cut:] = bin_img[128:384, cut:]

    w_result = find_vert_width_ROI(l_part, 20, False, True, 5)
    width = get_mean_diff(*w_result)
    g_result = find_gap_ROI(l_part, r_part)
    gap = get_mean_diff(*g_result)

    return w_result, width, g_result, gap
