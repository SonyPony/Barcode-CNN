# coding=utf-8
import os
import imageio
from random import randint
from math import ceil, floor
from PIL import Image
import numpy as np
import cv2 as cv
from json import loads

SUBDIR = "val"
PATH = "dataset/original/{}".format(SUBDIR)
END = len(os.listdir("{}".format(PATH))) / 2 - 1#498
OUT_PATH="dataset/restructurized/{}".format("08")

for i, filename in enumerate(os.listdir("{}/JPEGImages".format(PATH))):
    print("Start {}".format(filename))
    #img = cv.imread("{}/{:06}.jpg".format(PATH, i))
    img = cv.imread("{}/JPEGImages/{}".format(PATH, filename))

    mask = cv.imread("{}/SegmentationClass/{}".format(PATH, filename.replace(".jpg", ".png")))
    mask = mask[..., 2]
    mask[mask >= 10] = 255
    mask[mask < 10] = 0
    # load bbox corner points
    """with open("{}/{:06}.txt".format(PATH, i), "r+") as f:
        points = np.array(loads(f.read())["points"], dtype=int)

    # create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, points=points, color=255)"""

    cv.imwrite("{dir}/data/{filename:06}.jpg".format(dir=OUT_PATH, filename=i), img)
    cv.imwrite("{dir}/gt/{filename:06}.png".format(dir=OUT_PATH, filename=i), mask)
