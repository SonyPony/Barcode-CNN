# coding=utf-8
import skimage
import numpy as np


def add_noise(img, noise_count=1):
    for _ in range(noise_count):
        img = (skimage.util.random_noise(img, mode="poisson") * 255).astype(np.uint8)
    return img