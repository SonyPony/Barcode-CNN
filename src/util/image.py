# coding=utf-8
from PIL import Image
import numpy as np


def resize(img, new_size):
    p_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
    p_img = p_img.resize(new_size, Image.ANTIALIAS)

    img = np.asarray(p_img)
    return img