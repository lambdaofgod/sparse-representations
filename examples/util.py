import numpy as np


def threshold(x, threshold_value):
    return x * (np.abs(x) >= threshold_value)


def get_psnr(x, x_estimated):
    dynamic_range = x.max() - x.min()
    error = np.mean((x - x_estimated) ** 2)
    psnr_value = dynamic_range ** 2 / error
    return 10 * np.log10(psnr_value)