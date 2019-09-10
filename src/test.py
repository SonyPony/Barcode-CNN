# coding=utf-8
import numpy as np
import cv2 as cv

p1 = np.array([[978.0481,  435.45605],
 [900.2881,  528.76807],
 [813.60004, 228.96   ],
 [895.79535, 524.62085]]).astype(np.float32)

p3 = np.array([[979.0481,  435.45605],
 [901.2881,  528.76807],
 [813.60004, 228.96   ],
 [896.79535, 524.62085]]).astype(np.float32)

p2 = np.array([[979.7761,  435.45605],
 [902.0161,  528.76807],
 [815.04004, 228.96   ],
 [897.8689,  524.62085]]).astype(np.float32)

M = cv.getPerspectiveTransform(p1, p3)

S = np.array([
    [2, 3],
    [11, 6,],
[11, 6],
[11, 6],
])

S = cv.convertPointsToHomogeneous(S)
print("-"* 50)

print(M)

print("-"* 30)
print(np.dot(S, np.transpose(M)))