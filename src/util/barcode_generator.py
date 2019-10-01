# coding=utf-8
from typing import Tuple, Optional

from .color import random_bright_color
from barcode.writer import ImageWriter
import numpy as np
import barcode as bc
from PIL import Image


def random_barcode(rotation_angle, **kwargs):
    ean = bc.get('ean13')
    code = "".join(map(str, np.random.randint(10, size=12)))
    writer = ImageWriter()

    p_i = ean(code, writer=writer).render(writer_options=kwargs)

    # add baccode mask
    barcode = np.asarray(p_i)
    mask = np.ones(barcode.shape[:2], dtype=np.uint8) * 255
    barcode = np.dstack((barcode, mask))

    # rotate
    p_i = Image.fromarray(barcode)
    p_i = p_i.rotate(rotation_angle, expand=True)

    result = np.asarray(p_i)

    return result[..., :3], result[..., 3]



def random_barcode_with_bg(
        size_ratio: float=1,
        rotation_angle=0,
        color=None,
        **kwargs
):
    gen_barcode, mask = random_barcode(rotation_angle=rotation_angle, **kwargs)

    # add baccode mask
    #mask = np.ones(gen_barcode.shape[:2], dtype=np.uint8) * 255
    gen_barcode = np.dstack((gen_barcode, mask))

    # rotate
    """p_gen_barcode = Image.fromarray(gen_barcode)
    p_gen_barcode = p_gen_barcode.rotate(rotation_angle, expand=True)
    gen_barcode = np.asarray(p_gen_barcode)"""

    size = int(max(gen_barcode.shape) * (1. + size_ratio))
    size = np.array((size, size))

    # init result
    result = np.zeros(shape=(*size, 4), dtype=np.uint8)
    if not color:
        result[...] = (*random_bright_color(), 0)
    else:
        result[...] = (*color, 0)

    barcode_size = np.array(gen_barcode.shape[:2])
    offsets = ((size - barcode_size) / 2.).astype(int)

    # alpha matting of barcode
    mask = gen_barcode[..., 3] / 255.
    target_area = result[offsets[0]: offsets[0] + barcode_size[0], offsets[1]: offsets[1] + barcode_size[1]]
    target_area[...] = (1. - mask[..., None]) * target_area + mask[..., None] * gen_barcode

    return result[..., :3], result[..., 3]


def compose_barcode_with_bg(barcode: np.array, background: np.array, barcode_mask: np.array, translate_vector: Optional[Tuple]=None):
    barcode_norm_mask = barcode_mask / 255.
    barcode = np.dstack((barcode, barcode_mask))

    bg_size = np.array(background.shape[:2])
    barcode_size = np.array(barcode.shape[:2])

    offsets = np.array(translate_vector) if translate_vector else ((bg_size - barcode_size) / 2.).astype(int)
    assert np.all((barcode_size + offsets) <= np.array(background.shape[:2]))

    # blending
    result = np.zeros(shape=(*background.shape[:2], 4), dtype=np.uint8)
    result[..., :3] = background

    target_area = result[offsets[0]: offsets[0] + barcode_size[0], offsets[1]: offsets[1] + barcode_size[1]]
    target_area[...] = (1. - barcode_norm_mask[..., None]) * target_area \
                       + barcode_norm_mask[..., None] * barcode

    return result[..., :3], result[..., 3]