#!/usr/bin/env python

import torch

import numpy as np
from PIL import Image
import pandas
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


def decode(data: Union[str, List[int]], size: Tuple[int, int]=(768, 768)) -> torch.Tensor:
    if isinstance(data, str):
        data = [int(i) for i in data.split()]
    assert len(data) % 2 == 0, "dataが2の倍数でありません"
    assert len(data) == 0 or data[-1] <= np.prod(size), "dataの末尾が画像の範囲を超えています"

    img = torch.zeros(np.prod(size))
    for start, end in [data[i:i+2] for i in range(0, len(data), 2)]:
        for i in range(start, start + end):
            img[i-1] = 1
    return img.view(size).t()


def metric_to_img(x):
    if x is np.nan:
        return False, None
    else:
        return True, decode(x)


def show(img: torch.Tensor):
    if img.size()[0] == 3:
        img = img.numpy().transpose([1, 2, 0])
    plt.imshow(img)
    plt.pause(.01)


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"

    if any([isinstance(i, torch.Tensor) for i in batch]):
        return torch.stack([i for i in batch if i is not None], 0, out=None)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif all([i is None for i in batch]):
        return torch.empty(0)

    raise TypeError((error_msg.format(type(batch[0]))))
