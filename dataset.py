#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from util import decode
import pandas
import numpy as np

import os
from PIL import Image
import glob
import pickle


class SatelliteImages(Dataset):
    processed_folder = "input"
    train_dir = "train"
    train_csv = "train_ship_segmentations.csv"
    test_dir = "test"
    test_csv = "sample_submission.csv"

    train_pickle = "train_data.pickle"

    def __init__(self, root: str='.', train: bool=True, transform=None, bbox_transform=None, mask_transform=None,
                 on_server: bool=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.bbox_transform = bbox_transform
        self.mask_transform = mask_transform
        self.train = train

        if self.train:
            train_csv = pandas.read_csv(os.path.join(self.root, self.processed_folder, self.train_csv))

            if os.path.isfile(self.train_pickle):
                with open(self.train_pickle, "rb") as f:
                    self.data = pickle.load(f)
            else:
                train_data = {}
                for image_id, encoded_pixel in zip(train_csv["ImageId"], train_csv["EncodedPixels"]):
                    if encoded_pixel is np.nan:
                        train_data[image_id] = ()
                    else:
                        train_data[image_id] = (*train_data.get(image_id, ()), tuple(int(num) for num in encoded_pixel.split()))
                self.data = tuple(train_data.items())

                with open(self.train_pickle, "wb") as f:
                    pickle.dump(self.data, f)

            if not on_server:
                    all_train_data = dict(self.data)
                    train_data = {}
                    for file in [os.path.basename(i) for i in
                                 glob.glob(os.path.join(self.root, self.processed_folder, self.train_dir, "*"))]:
                        train_data[file] = all_train_data[file]
                    self.data = tuple(train_data.items())
        else:
            self.data = tuple(pandas.read_csv(os.path.join(self.root, self.processed_folder, self.test_csv))["ImageId"])

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index]
            img = Image.open(os.path.join(self.root, self.processed_folder, self.train_dir, img))
            bbox, mask = decode(target)
        else:
            img = self.data[index]
            img = Image.open(os.path.join(self.root, self.processed_folder, self.test_dir, img))

        if self.transform is not None:
            img = self.transform(img)

        if self.train and self.bbox_transform is not None:
            bbox = self.bbox_transform(bbox)

        if self.train and self.mask_transform is not None:
            mask = self.mask_transform(mask)

        if self.train:
            return img, bbox, mask
        else:
            return img

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = SatelliteImages(".")
    img = dataset[0]
