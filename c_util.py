#!/usr/bin/env python
from typing import List, Union, Tuple
import chainer
import numpy as np


def decode(data: Union[str, List[int], float], size: Tuple[int, int]=(768, 768)) -> np.ndarray:
    if data is np.nan:
        return np.zeros(size)
    if isinstance(data, str):
        data = [int(i) for i in data.split()]
    assert len(data) % 2 == 0, "dataが2の倍数でありません"
    assert len(data) == 0 or data[-1] <= np.prod(size), "dataの末尾が画像の範囲を超えています"

    img = np.zeros(np.prod(size))
    for start, end in [data[i:i+2] for i in range(0, len(data), 2)]:
        for i in range(start, start + end):
            img[i-1] = 1
    return np.reshape(img, size).transpose([1, 0])


class Normalize(object):
    def __init__(self, mean, std):
        assert len(mean) == len(std), "meanとstdの数が異なります"
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray):
        img = img / 255.
        for i, (m, s) in enumerate(zip(self.mean, self.std)):
            img[i] = (img[i] - m) / s
        return img
