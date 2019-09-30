# coding=utf-8
import random
import colorsys


def random_bright_color():
    h, s, l = random.random(), 0.5 + random.random() / 2., 0.8 + random.random() / 5.
    return tuple(map(lambda x: round(x * 255), colorsys.hls_to_rgb(h, l, s)))