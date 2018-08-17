#!/usr/bin/env python

import chainer
import chainer.functions as F
import numpy as np

import pandas

import os
from PIL import Image


class SatelliteImages(object):
    processed_folder = "input"
    train_dir = "train"
    train_csv = "train_ship_segmentations.csv"
    test_dir = "test"
    test_csv = "sample_submission.csv"

    def __init__(self, root: str, train: bool=True, transform=np.asarray, target_transform=None, download: bool=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data = pandas.read_csv(os.path.join(self.root, self.processed_folder, self.train_csv))
            self.train_data = list(zip(self.train_data["ImageId"], self.train_data["EncodedPixels"]))
        else:
            self.test_data = list(pandas.read_csv(os.path.join(self.root, self.processed_folder, self.test_csv))["ImageId"])

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index]
            img = Image.open(os.path.join(self.root, self.processed_folder, self.train_dir, img))
        else:
            img = self.test_data[index]
            img = Image.open(os.path.join(self.root, self.processed_folder, self.test_dir, img))

        if self.transform is not None:
            img = self.transform(img)

        if self.train and self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target
        else:
            return img

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SubSatelliteImages(SatelliteImages):
    def __init__(self, root: str, train: bool=True, transform=np.asarray, target_transform=None, download: bool=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        import glob
        if self.train:
            files = [os.path.basename(i) for i in glob.glob(os.path.join(self.root, self.processed_folder, self.train_dir, "*"))]
            self.train_data = pandas.read_csv(os.path.join(self.root, self.processed_folder, self.train_csv))
            self.train_data = list(filter(lambda x: x[0] in files, zip(self.train_data["ImageId"], self.train_data["EncodedPixels"])))
        else:
            self.test_data = [os.path.basename(i) for i in glob.glob(os.path.join(self.root, self.processed_folder, self.test_dir, "*"))]


if __name__ == "__main__":
    dataset = SatelliteImages(".")
    img = dataset[0]
