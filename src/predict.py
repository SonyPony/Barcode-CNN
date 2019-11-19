# coding=utf-8
import inspect

import torch
import argparse
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
import model as zoo
from torchvision import transforms

from util.filter import sobel_gradients
from util.image_pyramid import image_pyramid_scales
from torch.autograd import Variable
import torch.cuda
import matplotlib as mpl
from time import time

from util.time_measure import measure_time

plt.axis('off')
mpl.rcParams['figure.dpi'] = 300
models = dict(inspect.getmembers(zoo, lambda x : inspect.isclass(x) and issubclass(x, nn.Module)))

parser = argparse.ArgumentParser()
parser.add_argument("--input", action="store", required=True)
parser.add_argument("--out", action="store", required=True)
parser.add_argument("--pnet_type", action="store", required=True)
parser.add_argument("--rnet_type", action="store", required=True)
parser.add_argument("--onet_type", action="store", required=True)

parser.add_argument("--pnet_model", action="store", required=True)
parser.add_argument("--rnet_model", action="store", required=True)
parser.add_argument("--onet_model", action="store", required=True)

parser.add_argument("--pnet_grayscale", action="store", type=int, default=0)
parser.add_argument("--rnet_grayscale", action="store", type=int, default=0)
parser.add_argument("--onet_grayscale", action="store", type=int, default=0)
parser.add_argument("--onet_gradient", action="store", type=int, default=0)

args = parser.parse_args()

OUT_DIR = args.out
PNET_MODEL_PATH = args.pnet_model
RNET_MODEL_PATH = args.rnet_model
ONET_MODEL_PATH = args.onet_model

GRAY_PNET_INPUT = args.pnet_grayscale
GRAY_RNET_INPUT = args.rnet_grayscale
GRAY_ONET_INPUT = args.onet_grayscale

PNET_TYPE = args.pnet_type
RNET_TYPE = args.rnet_type
ONET_TYPE = args.onet_type

ONET_GRADIENT = args.onet_gradient


INPUT_PATH = args.input
#INPUT_PATH = "../dataset/restructurized/03/data/000005.jpg"
#INPUT_PATH = "../dataset/IMG_2152.jpg"
#INPUT_PATH = "/".join((OUT_DIR, "office5.jpg"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_gradients(img):
    return ((sobel_gradients(cv.cvtColor(img, cv.COLOR_BGR2GRAY)) > 120) * 255).astype(np.uint8)

def grayscale(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray = np.dstack((gray, gray, gray))

    return gray

def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.
        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def _preprocess(img):
    """Preprocessing step before feeding the network.
    Arguments:
        img: a float numpy array of shape [h, w, c].
    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255. - 0.5
    return img

def get_image_boxes(bounding_boxes, img, size=24, grads=None):
    """Cut out boxes from the image.
    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.
    Returns:
        a float numpy array of shape [n, 3, size, size].
    """

    use_grads = not (grads is None)
    print("Use gradients:", use_grads)
    channel_count = 4 if use_grads else 3

    num_boxes = len(bounding_boxes)
    height, width = img.shape[:2]

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, channel_count, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], channel_count), 'uint8')

        img_array = np.asarray(img, 'uint8')
        if use_grads:
            img_array = np.stack((img_array, grads))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] =\
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.ANTIALIAS)
        #img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline='blue')

    return img_copy

def run_pnet(model, img, scale, threshold=0.6):
    STRIDE = 2
    WIN_SIZE = 24

    h, w = img.shape[:2]

    if GRAY_PNET_INPUT:
        img = grayscale(img)

    img = Image.fromarray(img).resize((math.ceil(w * scale), math.ceil(h * scale)), Image.ANTIALIAS)
    img = transforms.ToTensor()(np.asarray(img, dtype=np.float32)) / 255. - 0.5
    input = img.unsqueeze(dim=0)

    offsets, labels = model(input.to(dev))
    offsets = offsets.cpu().detach().squeeze(dim=0).numpy()
    probs = F.softmax(labels).cpu().detach().squeeze(dim=0).numpy()[1]

    indices = np.where(probs > threshold)
    if indices[0].size == 0:
        return np.array([])

    #print(indices[0].shape)
    tx1, ty1, tx2, ty2 = [offsets[i, indices[0], indices[1]] for i in range(4)]
    #print(tx1.shape)
    offsets = np.array([tx1, ty1, tx2, ty2])
    #offsets = np.zeros(offsets.shape, dtype=offsets.dtype)
    score = probs[indices[0], indices[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    coeff = 0
    bounding_boxes = np.vstack([
        np.round((STRIDE * (indices[1]) + coeff * tx1) / scale),
        np.round((STRIDE * (indices[0]) + coeff * ty1) / scale),
        np.round((STRIDE * (indices[1]) - coeff * tx2 + WIN_SIZE) / scale),
        np.round((STRIDE * (indices[0]) - coeff * ty2 + WIN_SIZE) / scale),
        score, offsets
    ])
    # why one is added?

    return bounding_boxes.T


model = models[PNET_TYPE]()
model.load_state_dict(torch.load(PNET_MODEL_PATH)["weights"])
model.eval()
model = model.to(dev)

img = np.asarray(Image.open(INPUT_PATH))
grads = compute_gradients(img)[..., np.newaxis]

# build pyramid

with measure_time(print_format="Scales: {:.4f}s"):
    scales = image_pyramid_scales(img, factor=0.707, search_region=24, min_object_size=24)
print('scales:', ['{:.2f}'.format(s) for s in scales])


bounding_boxes = []

# run P-Net on different scales
with measure_time(print_format="Predict PNet: {:.4f}s"):
    for s in scales:
        #boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        boxes = run_pnet(model=model, img=img, scale=s, threshold=0.6)
        if boxes.size != 0:
            bounding_boxes.append(boxes)

# collect boxes (and offsets, and scores) from different scales
with measure_time(print_format="Stack RNet input: {:.4f}s"):
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)
print('number of bounding boxes:', len(bounding_boxes))


res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "1_p_net.png")))
#plt.imshow(res)
#plt.show()

# TODO use?

with measure_time(print_format="NMS + calibrate: {:.4f}s"):
    keep = nms(bounding_boxes, 0.7)
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

# use offsets predicted by pnet to transform bounding boxes

# shape [n_boxes, 5]

res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "1_p_net_nms.png")))

bounding_boxes = convert_to_square(bounding_boxes)
bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
print('number of bounding boxes:', len(bounding_boxes))

res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "1_p_pnet_nms_square.png")))
plt.imshow(res)
plt.show()

rnet = models[RNET_TYPE]()
rnet.load_state_dict(torch.load(RNET_MODEL_PATH)["weights"])
rnet.eval()
rnet = rnet.to(dev)

with measure_time(print_format="Predict RNet: {:.4f}s"):
    img_boxes = get_image_boxes(bounding_boxes, grayscale(img) if GRAY_RNET_INPUT else img, size=24)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)

    offsets, probs = rnet(img_boxes.to(dev))
    offsets = offsets.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = probs.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > 0.7)[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

print('number of bounding boxes:', len(bounding_boxes))
res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "2_r_net.png")))
plt.imshow(res)
plt.show()

with measure_time(print_format="NMS + calibrate: {:.4f}s"):
    keep = nms(bounding_boxes, 0.7)
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    res.save("/".join((OUT_DIR, "2_r_net_nms.png")))
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
print('number of bounding boxes:', len(bounding_boxes))

res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
plt.imshow(res)
plt.show()

#plt.savefig("out.png")

onet = models[ONET_TYPE]()
onet.load_state_dict(torch.load(ONET_MODEL_PATH)["weights"])
onet.eval()
onet = onet.to(dev)

with measure_time(print_format="ONet predict: {:.4f}s"):
    grads = grads if ONET_GRADIENT else None
    img_boxes = get_image_boxes(bounding_boxes, grayscale(img) if GRAY_ONET_INPUT else img, size=48, grads=grads)
    img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    output = onet(img_boxes.to(dev))
    offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > 0.8)[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

# compute landmark points
width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]

print('number of bounding boxes:', len(bounding_boxes))

res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "3_o_net.png")))
plt.imshow(res)
plt.show()


with measure_time(print_format="NMS + calibrate: {:.4f}s"):
    keep = nms(bounding_boxes, 0.7, mode='min')

    bounding_boxes = bounding_boxes[keep]
    offsets = offsets[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    print('number of bounding boxes:', len(bounding_boxes))

res = show_bboxes(Image.open(INPUT_PATH), bounding_boxes)
res.save("/".join((OUT_DIR, "3_o_net_nms.png")))
plt.imshow(res)
plt.show()