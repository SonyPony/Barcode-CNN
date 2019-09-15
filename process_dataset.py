# coding=utf-8
import os
import imageio
from random import randint
from math import ceil, floor
from PIL import Image
import numpy as np
import cv2 as cv
from json import loads
from imgaug import augmenters as iaa

def get_file_index(dir):
    files = tuple(map(lambda x: int(x.split(".")[0]), os.listdir("{}".format(dir))))
    if not len(files):
        return 0

    return max(files) + 1

PATH = "dataset/val"
OUT_PATH="dataset/val_data_rnet"
END = len(os.listdir("{}/JPEGImages".format(PATH))) - 1#498
IDX_POS = get_file_index(dir="{}/positive".format(OUT_PATH))
IDX_NEG = get_file_index(dir="{}/negative".format(OUT_PATH))
IDX_PART = get_file_index(dir="{}/part".format(OUT_PATH))
WIN_SIZE = 24


"""for filename in os.listdir("{}/negative".format(OUT_PATH)):
    if not(".jpg" in filename):
        continue

    img = cv.imread("{}/negative/{}".format(OUT_PATH, filename))
    if img.shape[:2] != (WIN_SIZE, WIN_SIZE):
        print("Fuck ", filename, img.shape)
exit(0)"""

def iou_mask(mask):
    area = mask.shape[0] * mask.shape[1]
    count = np.sum(mask != 0)

    return count / float(area)


def slice_win(bbox, win_size, offset=(0, 0)):
    x, y, w, h = bbox
    v_offset, h_offset = (win_size - h) / 2, (win_size - w) / 2
    v_padding, h_padding = 0, 0
    if y < v_offset:
        v_padding = int(ceil(v_offset - y))
    if x < h_offset:
        h_padding = int(ceil(h_offset - x))


    return np.s_[
       v_padding + offset[1] + y - ceil(v_offset): v_padding + offset[1] + y + h + floor(v_offset),
       h_padding + offset[0] + x - ceil(h_offset): h_padding + offset[0] + x + w + floor(h_offset)
   ]

def crop_win(target, bbox, win_size, offset=(0, 0), pad=False):
    res = target[slice_win(bbox=bbox, win_size=win_size, offset=offset)]

    if res.shape[:2] != (win_size, win_size) and pad:
        if len(res.shape) > 2 and res.shape[2] > 1:
            res = np.pad(res, ((0, win_size - res.shape[0]), (0, win_size - res.shape[1]), (0, 0)), "edge")
        else:
            res = np.pad(res, ((0, win_size - res.shape[0]), (0, win_size - res.shape[1])), "edge")
        #res = np.pad(res, ((0, 0), (0, 0)), "edge")

    return res

def save_crop_with_gt(crop, crop_mask, idx, path):
    p_img = Image.fromarray(crop)
    p_img.save("{}/{:06}.jpg".format(path, idx))

    if crop_mask is not None:
        points = cv.findNonZero(crop_mask)
        bbox_x, bbox_y, bbox_w, bbox_h = cv.boundingRect(points)

        with open("{}/{:06}.txt".format(path, idx), "w+") as f:
            content = "{left_offset} {top_offset} {right_offset} {bottom_offset}".format(
                left_offset=bbox_x, top_offset=bbox_y,
                right_offset=WIN_SIZE - bbox_x - bbox_w, bottom_offset=WIN_SIZE - bbox_y - bbox_h
            )

            f.write(content)

#for i in range(END + 1):
for i, filename in enumerate(os.listdir("{}/JPEGImages".format(PATH))):
    print("Start {}".format(filename))
    #img = cv.imread("{}/{:06}.jpg".format(PATH, i))
    img = cv.imread("{}/JPEGImages/{}".format(PATH, filename))

    # load bbox corner points
    """with open("{}/{:06}.txt".format(PATH, i), "r+") as f:
        points = np.array(loads(f.read())["points"], dtype=int)

    # create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, points=points, color=255)"""

    mask = cv.imread("{}/SegmentationClass/{}".format(PATH, filename.replace(".jpg", ".png")))
    mask = mask[..., 2]
    #mask[mask < 250] = 0
    points = cv.findNonZero(mask)

    # resize
    x, y, w, h = cv.boundingRect(points)
    scale = (WIN_SIZE / max(w, h))
    scaled_bbox = np.round(np.array([x, y, w, h], dtype=np.float32) * scale).astype(int)
    scaled_size = tuple(np.round(np.array(img.shape[:2]) * scale).astype(int))[::-1] # need to reverse, because np is h, w, c

    p_img = Image.fromarray(cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2RGB), "RGB")
    p_mask = Image.fromarray(mask.astype(np.uint8), "L")

    p_img = p_img.resize(scaled_size, Image.ANTIALIAS)
    p_mask = p_mask.resize(scaled_size, Image.ANTIALIAS)

    img = np.asarray(p_img).copy()
    mask = np.asarray(p_mask).copy()
    mask[mask < 90] = 0

    # POSITIVE
    crop = crop_win(img, bbox=scaled_bbox, win_size=WIN_SIZE, pad=True)
    crop_mask = crop_win(mask, bbox=scaled_bbox, win_size=WIN_SIZE, pad=True)

    #if crop_mask.shape == (WIN_SIZE, WIN_SIZE):
    save_crop_with_gt(crop=crop, crop_mask=crop_mask, idx=IDX_POS, path="{}/positive".format(OUT_PATH))
    IDX_POS += 1
    #else:
    #    print("Skip positive")

    # PART
    base_IoU = iou_mask(crop_mask)
    current_IoU = 0
    s_w, s_h = scaled_size

    tries = 0
    print("     Doing Part")
    while base_IoU * 0.4 > current_IoU or current_IoU > base_IoU * 0.65:
        tries += 1
        x, y = randint(-min(7, scaled_bbox[0]), min(7, s_w - scaled_bbox[2] - scaled_bbox[0])), \
               randint(-min(7, scaled_bbox[1]), min(7, s_h - scaled_bbox[3] - scaled_bbox[1]))

        crop = crop_win(img, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)
        if tries == 30:
            print("     Skip part")
            break
        if crop.shape[:2] != (WIN_SIZE, WIN_SIZE):
            continue

        crop_mask = crop_win(mask, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)

        current_IoU = iou_mask(crop_mask)

    if tries < 30:
        save_crop_with_gt(crop=crop, crop_mask=crop_mask, idx=IDX_PART, path="{}/part".format(OUT_PATH))
        IDX_PART += 1

    # NEGATIVE
    for _ in range(3):
        print("     Doing Neg {}".format(_))
        current_IoU = 1.
        tries = 0

        while current_IoU > base_IoU * 0.3:
            tries += 1
            #x, y = randint(-(s_w - WIN_SIZE), s_w - WIN_SIZE), randint(-(s_h - WIN_SIZE), s_h - WIN_SIZE)
            x, y = randint(-scaled_bbox[0], s_w - scaled_bbox[2] - scaled_bbox[0]), \
                   randint(-scaled_bbox[1], s_h - scaled_bbox[3] - scaled_bbox[1])

            crop = crop_win(img, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)
            if tries == 30:
                print("     Skip negative {}".format(_))
                break

            if crop.shape[:2] != (WIN_SIZE, WIN_SIZE):
                continue

            crop_mask = crop_win(mask, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)
            current_IoU = iou_mask(crop_mask)

        if tries < 30:
            save_crop_with_gt(crop=crop, crop_mask=None, idx=IDX_NEG, path="{}/negative".format(OUT_PATH))
            IDX_NEG += 1

    #cv.namedWindow("d", cv.WINDOW_NORMAL)
    #cv.imshow("d", crop)
    #cv.waitKey()

    print("Processed {}/{}".format(i + 1, END + 1))
