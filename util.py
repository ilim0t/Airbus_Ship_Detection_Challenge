#!/usr/bin/env python

import torch

import numpy as np
from PIL import Image
import pandas
import matplotlib
import platform
if platform.system() != "Darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections

from typing import Tuple, Union, List

INPUT_DIR = "input/"
SEG_FILE = INPUT_DIR + "train_ship_segmentations.csv"


def loaf_img(path: str) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def encode(img: np.ndarray) -> str:
    img = img.squeeze()
    assert img.ndim == 2, "画像が二値ではありません"
    data = []
    for i, pixel in enumerate(img.transpose([1, 0]).flatten()):
        if pixel and len(data) % 2 == 0:
            data.append(i+1)
        elif not pixel and len(data) % 2 == 1:
            data.append(i+1 - data[-1])
    if len(data) % 2 != 2:
        data.append(img.size - data[-1] + 1)
    return " ".join([str(i) for i in data])


def decode(data: Tuple[Tuple[int]], size: Tuple[int, int]=(768, 768)) -> Tuple[np.ndarray]:
    if len(data) == 0:
        return np.full((1, 4), -1), np.full((1, *size), -1)
    # if isinstance(data, str):
    #     data = [int(i) for i in data.split()]
    result = []
    cord_result = []
    bbox, mask = np.zeros((len(data), 4)), np.zeros((len(data), *size))
    for i, obj in enumerate(data):
        assert len(obj) % 2 == 0, "dataが2の倍数でありません"
        assert len(obj) == 0 or obj[-1] <= np.prod(size), "dataの末尾が画像の範囲を超えています"
        for start, end in [obj[j:j + 2] for j in range(0, len(obj), 2)]:
            for k in range(start, start + end):
                # mask[i][(k - 1) % size[0]][(k - 1) // size[0]] = 1  # transpose ?
                mask[(i, *divmod(k - 1, size[0])[::-1])] = 1
        x, y = np.where(mask[i])
        bbox[i] = (np.array((min(y), min(x), max(y), max(x))))  # 順番は要検証
    return bbox, mask


def show(img: Union[torch.Tensor, np.ndarray]):
    if isinstance(img, list):
        img = sum(img)
    if torch.is_tensor(img):
        img = img.numpy()
    if img.shape[0] == 3:
        img = img.transpose([1, 2, 0])
    plt.imshow(img)
    plt.pause(.01)


def collate(batch, raw=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], collections.Sequence):
        if raw:
            return batch
        transposed = zip(*batch)
        return [collate(samples, True) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class Normalize(object):
    def __init__(self, mean, std):
        assert len(mean) == len(std), "meanとstdの数が異なります"
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img / 255.
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            img[i] = (img[i] - m) / s
        return img


# import six
#
# def concat_examples(batch, device=None, padding=None):
#     if len(batch) == 0:
#         raise ValueError('batch is empty')
#
#     first_elem = batch[0]
#
#     if isinstance(first_elem, tuple):
#         result = []
#         if not isinstance(padding, tuple):
#             padding = [padding] * len(first_elem)
#
#         for i in six.moves.range(len(first_elem)):
#             result.append(to_device(device, _concat_arrays(
#                 [example[i] for example in batch], padding[i])))
#
#         return tuple(result)
#
#     elif isinstance(first_elem, dict):
#         result = {}
#         if not isinstance(padding, dict):
#             padding = {key: padding for key in first_elem}
#
#         for key in first_elem:
#             result[key] = to_device(device, _concat_arrays(
#                 [example[key] for example in batch], padding[key]))
#
#         return result
#
#     else:
#         return to_device(device, _concat_arrays(batch, padding))
#
#
# def _concat_arrays(arrays, padding):
#     # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
#     # types such as int or float.
#     if not isinstance(arrays[0], numpy.ndarray) and\
#        not isinstance(arrays[0], cuda.ndarray):
#         arrays = numpy.asarray(arrays)
#     if padding is not None:
#         return _concat_arrays_with_padding(arrays, padding)
#
#     xp = cuda.get_array_module(arrays[0])
#     with cuda.get_device_from_array(arrays[0]):
#         return xp.concatenate([array[None] for array in arrays])
#
#
# def _concat_arrays_with_padding(arrays, padding):
#     shape = numpy.array(arrays[0].shape, dtype=int)
#     for array in arrays[1:]:
#         if numpy.any(shape != array.shape):
#             numpy.maximum(shape, array.shape, shape)
#     shape = tuple(numpy.insert(shape, 0, len(arrays)))
#
#     xp = cuda.get_array_module(arrays[0])
#     with cuda.get_device_from_array(arrays[0]):
#         result = xp.full(shape, padding, dtype=arrays[0].dtype)
#         for i in six.moves.range(len(arrays)):
#             src = arrays[i]
#             slices = tuple(slice(dim) for dim in src.shape)
#             result[(i,) + slices] = src
#
#     return result
