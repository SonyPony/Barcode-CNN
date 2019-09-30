# coding=utf-8
from random import randint
import os
from PIL import Image
import  numpy as np
from util.dataset import get_file_index


PATH = "C:/Users/Sony/Downloads/VOCdevkit/VOC2012/JPEGImages"
OUT_PATH="../../dataset/test2"
IDX = get_file_index(dir=OUT_PATH)
WIN_SIZE = 24


count = len(os.listdir(PATH))
for i, filename in enumerate(os.listdir(PATH)):
    p_img = Image.open("{}/{}".format(PATH, filename))

    #create crop
    for _ in range(2):
        crop = np.asarray(p_img)
        x, y = randint(0, crop.shape[1] - WIN_SIZE - 1), randint(0, crop.shape[0] - WIN_SIZE - 1)

        crop = crop[y: y+WIN_SIZE, x: x+WIN_SIZE]
        p_crop = Image.fromarray(crop)
        p_crop.save("{}/{:06}.jpg".format(OUT_PATH, IDX))
        IDX += 1

    # resize to thumbnail
    p_img = p_img.resize((WIN_SIZE, WIN_SIZE))
    p_img.save("{}/{:06}.jpg".format(OUT_PATH, IDX))

    print("Processed {}/{} - {:06}.jpg".format(i + 1, count, IDX))
    IDX += 1