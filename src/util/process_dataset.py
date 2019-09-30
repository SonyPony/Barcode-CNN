# coding=utf-8
import os
import imageio
from random import randint
from math import ceil, floor
from PIL import Image
import numpy as np
import cv2 as cv
from util.dataset import get_file_index

np.seterr(all="raise")
WIN_SIZE = 48



def save_crop_with_gt(crop, crop_mask, idx, path):
    p_img = Image.fromarray(crop)
    p_img.save("{}/{:06}.jpg".format(path, idx))

    if crop_mask is not None:
        points = cv.findNonZero(crop_mask)
        bbox_x, bbox_y, bbox_w, bbox_h = cv.boundingRect(points)

        with open("{}/{:06}.txt".format(path, idx), "w+") as f:
            content = "{left_offset:.6f} {top_offset:.6f} {right_offset:.6f} {bottom_offset:.6f}".format(
                left_offset=bbox_x / WIN_SIZE,
                top_offset=bbox_y / WIN_SIZE,
                right_offset=-(WIN_SIZE - bbox_x - bbox_w) / WIN_SIZE,
                bottom_offset=-(WIN_SIZE - bbox_y - bbox_h) / WIN_SIZE
            )

            f.write(content)

def process_negatives(PATH):
    OUT_PATH = "dataset/onet_train_data"
    END = len(os.listdir("{}".format(PATH))) - 1  # 498
    IDX_NEG = get_file_index(dir="{}/negative".format(OUT_PATH))

    for i, filename in enumerate(os.listdir("{}".format(PATH))):
        for scale in (1, 0.5, 0.25, 0.8, 0.15):
            for _ in range(24):
                print("Start {}".format(filename))
                angle = int(round(np.random.uniform(0, 360)))

                # img = cv.imread("{}/{:06}.jpg".format(PATH, i))
                img = cv.imread("{}/{}".format(PATH, filename))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                p_img = Image.fromarray(img, "RGB")
                p_img = p_img.resize((int(img.shape[1] * scale), int(img.shape[0] * scale)), Image.ANTIALIAS)

                img = np.asarray(p_img)

                s_h, s_w = img.shape[:2]

                x, y = randint(0, s_w - WIN_SIZE - 1), randint(0, s_h - WIN_SIZE - 1)

                # crop = crop_win(img, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)
                crop = img[y:y + WIN_SIZE, x:x + WIN_SIZE]
                crop = np.asarray(Image.fromarray(crop).rotate(angle, expand=False))
                save_crop_with_gt(
                    crop=crop, crop_mask=None, idx=IDX_NEG,
                    path="{}/negative".format(OUT_PATH)
                )
                IDX_NEG += 1



def process_dataset(PATH):
    SKIP_POSITIVE = False
    SKIP_PART = False
    SKIP_NEGATIVE = False
    #PATH = "dataset/restructurized/01"
    OUT_PATH="dataset/onet_val_data"
    END = len(os.listdir("{}/data".format(PATH))) - 1#498
    IDX_POS = get_file_index(dir="{}/positive".format(OUT_PATH))
    IDX_NEG = get_file_index(dir="{}/negative".format(OUT_PATH))
    IDX_PART = get_file_index(dir="{}/part".format(OUT_PATH))


    """for filename in os.listdir("{}/positive".format(OUT_PATH)):
        if not(".jpg" in filename):
            continue
    
        img = cv.imread("{}/positive/{}".format(OUT_PATH, filename))
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



    for rel_scale in (48, 47, 46, 45, 44, 43, 42, 41, 40, 39):
        print("SCALE {} {}".format(rel_scale, "-" * 30))
        print(rel_scale / WIN_SIZE)
        for i, filename in enumerate(os.listdir("{}/data".format(PATH))):
            print("Start {}".format(filename))
            angle = int(round(np.random.uniform(-15, 15)))
            angle = 0

            #img = cv.imread("{}/{:06}.jpg".format(PATH, i))
            img = cv.imread("{}/data/{}".format(PATH, filename))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            p_img = Image.fromarray(img, "RGB")
            p_img = p_img.rotate(angle)
            img = np.asarray(p_img)

            # load bbox corner points
            """with open("{}/{:06}.txt".format(PATH, i), "r+") as f:
                points = np.array(loads(f.read())["points"], dtype=int)
    
            # create mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv.fillConvexPoly(mask, points=points, color=255)"""

            print("{}/gt/{}".format(PATH, filename.replace(".jpg", ".png")))
            mask = cv.imread("{}/gt/{}".format(PATH, filename.replace(".jpg", ".png")), cv.IMREAD_GRAYSCALE)
            p_mask = Image.fromarray(mask, "L")
            p_mask = p_mask.rotate(angle)
            mask = np.asarray(p_mask)

            #mask = mask[..., 2]
            #mask[mask < 250] = 0
            points = cv.findNonZero(mask)

            # resize
            x, y, w, h = cv.boundingRect(points)
            scale = (WIN_SIZE / max(w, h)) * (rel_scale / WIN_SIZE)
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

            if not SKIP_POSITIVE:
                save_crop_with_gt(crop=crop, crop_mask=crop_mask, idx=IDX_POS, path="{}/positive".format(OUT_PATH))
            IDX_POS += 1
            base_IoU = iou_mask(crop_mask)
            current_IoU = 0
            s_w, s_h = scaled_size

            tries = 0
            print("     Doing Part")
            if not SKIP_PART:
                while 0.4 * base_IoU > current_IoU or current_IoU > 0.65 * base_IoU:
                    tries += 1
                    x, y = randint(-min(WIN_SIZE / 2, scaled_bbox[0]), min(WIN_SIZE / 2, s_w - scaled_bbox[2] - scaled_bbox[0])), \
                           randint(-min(WIN_SIZE / 2, scaled_bbox[1]), min(WIN_SIZE / 2, s_h - scaled_bbox[3] - scaled_bbox[1]))

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
            if not SKIP_NEGATIVE:
                for _ in range(3):
                    print("     Doing Neg {}".format(_))
                    current_IoU = 1.
                    tries = 0

                    while current_IoU > 0.3 * base_IoU:
                        tries += 1
                        #x, y = randint(-(s_w - WIN_SIZE), s_w - WIN_SIZE), randint(-(s_h - WIN_SIZE), s_h - WIN_SIZE)
                        """x, y = randint(-scaled_bbox[0], s_w - scaled_bbox[2] - scaled_bbox[0]), \
                               randint(-scaled_bbox[1], s_h - scaled_bbox[3] - scaled_bbox[1])"""
                        if tries == 30:
                            print("     Skip negative {}".format(_))
                            break
                        if s_w <= WIN_SIZE or s_h <= WIN_SIZE:
                            continue
                        x, y = randint(0, s_w - WIN_SIZE - 1), randint(0,  s_h - WIN_SIZE - 1)

                        #crop = crop_win(img, offset=(x, y), bbox=scaled_bbox, win_size=WIN_SIZE)
                        crop = img[y:y + WIN_SIZE, x:x + WIN_SIZE]

                        if crop.shape[:2] != (WIN_SIZE, WIN_SIZE):
                            continue

                        crop_mask = mask[y:y + WIN_SIZE, x:x + WIN_SIZE]
                        current_IoU = iou_mask(crop_mask)

                    if tries < 30:
                        save_crop_with_gt(
                            crop=crop, crop_mask=None, idx=IDX_NEG,
                            path="{}/negative".format(OUT_PATH)
                        )
                        IDX_NEG += 1

            print("Processed {}/{}".format(i + 1, END + 1))

process_negatives("dataset/original/negative")
exit(0)
for i in (8,):
    process_dataset("dataset/restructurized/{:02}".format(i))