# coding=utf-8
import os
from math import floor
from PIL import Image
import cv2 as cv
import numpy as np
from util.barcode_generator import random_barcode_with_bg, random_barcode, compose_barcode_with_bg
from util.dataset import get_file_index, save_crop_with_gt
import random
from util import add_noise, random_bright_color
import util.image as image
import scipy.ndimage as ndimage


static_barcode_options = {
    "format": "PNG",
    "dpi": 200
}

BG_PATH = "C:/Users/Sony/Downloads/VOCdevkit/VOC2012/JPEGImages"
BG_FILES = os.listdir(BG_PATH)
BG_COUNT = len(BG_FILES)
OUT_PATH="../../dataset/test3"
POSITIVE_PATH = "{}/positive".format(OUT_PATH)
PART_PATH = "{}/part".format(OUT_PATH)
POSITIVE_IDX = get_file_index(dir=POSITIVE_PATH)
PART_IDX = get_file_index(dir=PART_PATH)
COUNT = 20000
WIN_SIZE = 24
ANGLE_RANGE = 180
TRANSLATE = True


for i in range(COUNT):
    white_bg = random.random() < 0.1
    white_barcode_bg = random.random() < 0.1

    barcode_size_ratio = random.random() * 0.35 + 0.65
    barcode, barcode_mask = random_barcode(
        rotation_angle=random.randint(-ANGLE_RANGE, ANGLE_RANGE),
        module_height=random.random() * 28 + 4,
        quiet_zone=random.random() * 5 + 1,
        text_distance=random.random() * 3,
        background="white" if white_barcode_bg else "#{:02x}{:02x}{:02x}".format(*random_bright_color()),
        foreground="#{}".format(3 * "{:02x}".format(random.randint(0, 80))),
        **static_barcode_options,
    )
    barcode = add_noise(img=barcode, noise_count=3)

    # RESIZE to rations
    barcode_size = floor(WIN_SIZE * barcode_size_ratio)
    barcode = image.resize(barcode, (barcode_size, barcode_size))
    barcode_mask = image.resize(barcode_mask, (barcode_size, barcode_size))

    background = Image.open("{}/{}".format(BG_PATH, BG_FILES[random.randint(0, BG_COUNT - 1)]))

    # white background
    if white_bg:
        background = np.asarray(background)
        background = np.ones(background.shape, dtype=background.dtype) * 255

    if random.random() < 0.5 or white_bg:
        background = image.resize(background, new_size=(WIN_SIZE, WIN_SIZE))

    else:
        background = np.asarray(background)
        x, y = random.randint(0, background.shape[1] - WIN_SIZE - 1), random.randint(0, background.shape[0] - WIN_SIZE - 1)
        background = background[y: y+WIN_SIZE, x: x+WIN_SIZE]

    offset = None if not TRANSLATE else \
        (random.randint(0, WIN_SIZE - barcode_size), random.randint(0, WIN_SIZE - barcode_size))

    # CREATING PART
    if barcode_size_ratio > 0.7:
        base_iou = image.iou_mask(barcode_mask)
        current_iou = 0.

        part_result, part_result_mask = None, None
        dx, dy = 0, 0
        tries = 0
        while current_iou > 0.8 * base_iou or current_iou < base_iou * 0.5:
            dy, dx = random.randint(-WIN_SIZE // 3, WIN_SIZE // 3), random.randint(-WIN_SIZE // 3, WIN_SIZE // 3)
            part_result, part_result_mask = compose_barcode_with_bg(
                barcode=barcode,
                background=background,
                barcode_mask=barcode_mask,
                translate_vector=(dy, dx)
            )

            current_iou = image.iou_mask(part_result_mask)
            tries += 1
            if tries > 30:
                print("     Skip part")
                break

        # save part barcode
        if tries <= 30:
            save_crop_with_gt(
                crop=part_result,
                crop_mask=part_result_mask,
                idx=PART_IDX,
                path=PART_PATH,
                win_size=WIN_SIZE
            )

            PART_IDX += 1

    result, result_mask = compose_barcode_with_bg(
        barcode=barcode,
        background=background,
        barcode_mask=barcode_mask,
        translate_vector=offset if TRANSLATE else None
    )

    # save image with gt
    save_crop_with_gt(
        crop=result,
        crop_mask=result_mask,
        idx=POSITIVE_IDX,
        path=POSITIVE_PATH,
        win_size=WIN_SIZE
    )

    print("Generated {}/{} - {:06}.jpg".format(i + 1, COUNT, POSITIVE_IDX))
    POSITIVE_IDX += 1




