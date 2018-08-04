#!/usr/bin/env python

import numpy as np
from PIL import Image
import pandas
import matplotlib.pyplot as plt

from typing import Tuple, Union

INPUT_DIR = "input/"
SEG_FILE = INPUT_DIR + "train_ship_segmentations.csv"


def loaf_img(fname: str) -> np.ndarray:
    return np.asarray(Image.open(INPUT_DIR + fname))


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


def decode(data: str, size: Tuple[int, int]=(768, 768)) -> np.ndarray:
    data = [int(i) for i in data.split()]
    assert len(data) % 2 == 0, "dataが2の倍数でありません"
    assert len(data) == 0 or data[-1] <= np.prod(size), "dataの末尾が画像の範囲を超えています"

    img = np.zeros(np.prod(size)).astype(bool)
    for start, end in [data[i:i+2] for i in range(0, len(data), 2)]:
        for i in range(start, start + end):
            img[i-1] = True
    return img.reshape(size).transpose([1, 0])


def show(img: np.ndarray):
    plt.imshow(img)
    plt.pause(.01)


class seg(object):
    def __init__(self):
        self.df = pandas.read_csv(SEG_FILE)

    def __call__(self, index: int) -> Tuple[str, Union[float, str]]:
        return tuple(self.df.iloc[index])
