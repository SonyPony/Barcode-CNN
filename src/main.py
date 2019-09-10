# coding=utf-8
import os.path
import numpy as np
import cv2 as cv
from json import dumps

grayscale = lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY)
keypoint = lambda x: cv.KeyPoint(*x, 5)
keypoints2points = lambda x: np.array([kp.pt for kp in x])
match_idxs = lambda x, y: np.array([getattr(o, y) for o in x])
SKIP = 76
END = 179

def chroma_key(frame, keyed_color, threshold):
    frame = frame.astype(float) / 255.
    diff = np.zeros(frame.shape[:2], dtype=float)

    for i in range(3):
        diff += np.abs(frame[..., i] - keyed_color[i] / 255.)
    return diff < threshold


def save_frame_gt(dir, frame, points, i):
    cv.imwrite("{}/{:06}.jpg".format(dir, i), frame)
    #cv.imwrite("{}/{:06}.jpg".format(dir, i), frame)
    with open("{}/{:06}.txt".format(dir, i), "w+") as f:
        str_points = dumps({"points": np.squeeze(points).tolist()})
        f.write(str_points)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(80, 80), maxLevel=1, criteria=(cv.TERM_CRITERIA_COUNT, 30, 0.01))


# load video
FILE_DIR = "../dataset/04"
FILE_PATH = "{}/IMG_1836.mov".format(FILE_DIR)
USED_KEYPOINTS_COUNT = 80

cap = cv.VideoCapture(FILE_PATH)
orb = cv.ORB_create()

i = 0
_, old_frame = cap.read()
while i < SKIP:
    _, old_frame = cap.read()
    i += 1
#if i < SKIP:
#    print("Skip {}".format(i))
#_, old_frame = cap.read()

if not os.path.exists("{}/first_frame.jpg".format(FILE_DIR)):
    cv.imwrite("{}/first_frame.jpg".format(FILE_DIR), old_frame)
    print("Re-run")
    exit(0)

points = np.array([[[1030.25, 682.61767578125]], [[1133.7520751953125, 623.44921875]], [[1160.605224609375, 669.1279296875]], [[1056.005615234375, 721.7789916992188]]], dtype=np.float32)
"""points = np.array([
    [[755, 929]],   # TL
    [[860, 859]],   # TR
    [[891, 909]],   # BR
    [[786, 980]],   # BL
], dtype=np.float32)"""
orig_points = np.squeeze(points, axis=1)

kp = orb.detect(grayscale(old_frame), None)
#kp = list(map(keypoint, np.squeeze(points, axis=1)))
kp, des = orb.compute(grayscale(old_frame), kp)


save_frame_gt(dir=FILE_DIR, frame=old_frame, points=points, i=i)

"""K = np.array([42, 129, 65])
mask = chroma_key(old_frame, K, threshold=0.30)
old_frame[mask, :] = 255
cv.imshow("ff", old_frame)
cv.waitKey()

exit(0)"""

while cap.isOpened():
    _, frame = cap.read()

    i += 1
    if i >= END:
        break

    tr_points, sr, err = cv.calcOpticalFlowPyrLK(grayscale(old_frame), grayscale(frame), points, None, **lk_params)
    old_frame = frame
    points = tr_points

    print(i)


    """tr_kp, tr_des = orb.detectAndCompute(grayscale(frame), None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des, tr_des)
    matches = sorted(matches, key=lambda x: x.distance)[:USED_KEYPOINTS_COUNT]

    matched_orig_kp = keypoints2points(kp)[match_idxs(matches, "queryIdx")].astype(np.float32)
    matched_tracked_kp = keypoints2points(tr_kp)[match_idxs(matches, "trainIdx")].astype(np.float32


    H, _ = cv.findHomography(matched_orig_kp.reshape(-1, 2, 1), matched_tracked_kp.reshape(-1, 2, 1), method=cv.RANSAC, ransacReprojThreshold=0.12)
    homo_orig_points = cv.convertPointsToHomogeneous(orig_points)


    homo_tr_points = np.dot(homo_orig_points, np.transpose(H))
    tr_points = cv.convertPointsFromHomogeneous(homo_tr_points)"""

    save_frame_gt(dir=FILE_DIR, frame=frame, points=tr_points, i=i)

    frame = cv.polylines(frame, pts=[tr_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
    """matches = cv.drawMatches(old_frame, kp, frame, tr_kp, matches[:USED_KEYPOINTS_COUNT], flags=2, outImg=None)

    matches = cv.resize(matches, fx=0.5, fy=0.5, dsize=(0, 0))
    frame = cv.resize(frame, fx=0.5, fy=0.5, dsize=(0, 0))
    cv.imshow("m", matches)"""

    cv.imwrite("{}/check/{:06}.jpg".format(FILE_DIR, i - 1), frame)
    #cv.imshow("win", frame)
    #cv.waitKey()

cap.release()
cv.destroyAllWindows()