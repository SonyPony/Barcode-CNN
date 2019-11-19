# coding=utf-8
import numpy as np
from scipy.signal import convolve2d


def generic_filter(source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Generic version for 2D convolve filtration specified by filter matrix.
    """
    return convolve2d(source, matrix)

def sobel_gradients_vectors(source: np.ndarray) -> np.ndarray:
    """
    Computes partial derivations to detect angle gradients.
    """
    grad_x = generic_filter(source, np.matrix([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]]
    ))

    grad_y = generic_filter(source, np.matrix([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]
    ))

    return np.array((grad_y, grad_x))

def sobel_gradients(source: np.ndarray) -> np.ndarray:
    """
    Computes partial derivations to detect angle gradients.
    """
    grad_y, grad_x = sobel_gradients_vectors(source)

    grads = np.hypot(grad_y, grad_x)
    return grads[1:-1:, 1:-1:]