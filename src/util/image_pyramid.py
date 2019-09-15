# coding=utf-8
import numpy as np
from PIL import Image


def image_pyramid_scales(img, factor, search_region, min_object_size):
    size = min(*img.shape[:2])
    biggest_size_ratio = search_region / min_object_size
    size *= biggest_size_ratio

    scales = []

    factor_power = 0
    while size > search_region:
        scales.append(biggest_size_ratio * factor ** factor_power)
        size *= factor
        factor_power += 1

    return scales