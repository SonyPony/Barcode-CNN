# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import os
from skimage import io


class MASDataset(Dataset):
    def __init__(self, directories, transform=None, grayscale=False):
        self._dir = (directories, ) if isinstance(directories, str) else directories
        self._transform = transform
        self._grayscale = grayscale

        self._data_path = {
            "positive": list(),
            "part": list(),
            "negative": list()
        }

        for sub_folder in self._data_path.keys():
            for single_dir in self._dir:
                paths = filter(lambda x: not x.endswith(".txt"), map(
                    lambda x: "/".join((single_dir, sub_folder, x)),
                    os.listdir(os.path.join(self._dir, sub_folder))
                ))

                self._data_path[sub_folder].extend(paths)


        self._positive_len = len(self._data_path["positive"])
        self._part_len = len(self._data_path["part"])
        self._negative_len = len(self._data_path["negative"])

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

        #image = io.imread("{}/{}/{:06}.jpg".format(self._dir, subdir, idx))
        image = io.imread(self._data_path[subdir][idx])
        bbox = (0., 0., 0., 0.)

        if subdir != "negative":
            # with open("{}/{}/{:06}.txt".format(self._dir, subdir, idx), "r+") as f:
            with open(self._data_path[subdir][idx].replace(".jpg", ".txt"), "r+") as f:
                bbox = tuple(map(float, f.read().split(" ")))

        # TODO transform bbox?
        if self._grayscale:
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

