# coding=utf-8
import os
from PIL import Image
import cv2 as cv


def save_crop_with_gt(crop, crop_mask, idx, path, win_size):
    p_img = Image.fromarray(crop)
    p_img.save("{}/{:06}.jpg".format(path, idx))

    if crop_mask is not None:
        points = cv.findNonZero(crop_mask)
        bbox_x, bbox_y, bbox_w, bbox_h = cv.boundingRect(points)

        with open("{}/{:06}.txt".format(path, idx), "w+") as f:
            content = "{left_offset:.6f} {top_offset:.6f} {right_offset:.6f} {bottom_offset:.6f}".format(
                left_offset=bbox_x / win_size,
                top_offset=bbox_y / win_size,
                right_offset=-(win_size - bbox_x - bbox_w) / win_size,
                bottom_offset=-(win_size - bbox_y - bbox_h) / win_size
            )

            f.write(content)

def get_file_index(dir):
    files = tuple(map(lambda x: int(x.split(".")[0]), os.listdir("{}".format(dir))))
    if not len(files):
        return 0

    return max(files) + 1