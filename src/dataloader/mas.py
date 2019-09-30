# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import os
from skimage import io


class MASDataset(Dataset):
    def __init__(self, directory, transform=None):
        self._dir = directory
        self._transform = transform

        self._positive_len = len(os.listdir("{}/positive".format(self._dir))) // 2
        self._part_len = len(os.listdir("{}/part".format(self._dir))) // 2
        self._negative_len = len(os.listdir("{}/negative".format(self._dir)))

    def __len__(self):
        return self._positive_len + self._part_len + self._negative_len

    def __getitem__(self, idx):
        subdir = "positive"

        if idx >= self._positive_len + self._part_len:
            subdir = "negative"
            idx = idx - self._positive_len - self._part_len

        elif idx >= self._positive_len:
            subdir = "part"
            idx = idx - self._positive_len

        image = io.imread("{}/{}/{:06}.jpg".format(self._dir, subdir, idx))
        bbox = (0., 0., 0., 0.)

        if subdir != "negative":
            with open("{}/{}/{:06}.txt".format(self._dir, subdir, idx), "r+") as f:
                bbox = tuple(map(float, f.read().split(" ")))

        # TODO transform bbox?
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = np.dstack((image, image, image))

        image = image / 255. - 0.5
        if self._transform:
            image = self._transform(image)

        return {
            "label": torch.tensor(subdir != "negative"),
            "bbox": torch.tensor(bbox),
            "image": image.float()
        }

