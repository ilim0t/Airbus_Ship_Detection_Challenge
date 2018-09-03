#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split

import argparse
import json
import numpy as np
from typing import Tuple

from dataset import SatelliteImages
from util import decode, show


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, bbox, mask) in enumerate(train_loader): # data, (bboxs, masks)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, bbox, mask)
        loss = 0
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, eval_dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_dataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(eval_dataset.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(eval_dataset.dataset),
        100. * correct / len(eval_dataset.dataset)))


import collections
import re
from torch._six import string_classes, int_classes


def collate(batch):
    transposed = tuple(zip(*batch))

    out = None
    if torch.utils.data.dataloader._use_shared_memory:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in transposed[0]])
        storage = transposed[0][0].storage()._new_shared(numel)
        out = transposed[0][0].new(storage)

    return torch.stack(transposed[0], 0, out=out), transposed[1], transposed[2]


class MaskRCNNTrain(nn.Module):
    def __init__(self, mask_rcnn):
        super(MaskRCNNTrain, self).__init__()
        self.mask_rcnn = mask_rcnn
        self.n_sample = 128
        self.pos_ratio = 0.25

    def forward(self, x: torch.Tensor, bbox: Tuple[np.ndarray], label: Tuple[np.ndarray]):
        # (ymin, xmin, ymax, xmax)
        # (y, x, h, w)
        # の順
        # [bg, fg]

        # Backbone, Extractor
        features = self.mask_rcnn.extractor(x)

        assert x.size(2) % features.size(2) == x.size(3) % features.size(3) == 0
        # RPN
        anchor = self.mask_rcnn.enumerate_shifted_anchor(x.shape[2:], features.shape[2:])

        h = F.relu(self.mask_rcnn.intermediate(features))

        # -1 -> features.size(2) * features.size(3) * self.n_anchor
        score = self.mask_rcnn.cls(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 2))
        offset = self.mask_rcnn.region(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 4))

        for i, (s, o, b, l) in enumerate(zip(score, offset, bbox, label)):
            roi = self.mask_rcnn.rpn(anchor, o, s, x.size(2), x.size(3))
            self.target_roi(roi, b, l)

    def target_roi(self, roi, bbox, label):
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        roi = roi.detach().numpy()

        roi = np.concatenate((roi, bbox))

        tl = np.maximum(roi[:, None, :2], bbox[:, :2])
        br = np.minimum(roi[:, None, 2:], bbox[:, 2:])

        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(roi[:, 2:] - roi[:, :2], axis=1)
        area_b = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        iou = area_i / (area_a[:, None] + area_b - area_i)


class MaskRCNN(nn.Module):
    def __init__(self, class_num=1):
        super(MaskRCNN, self).__init__()
        self.extractor = models.vgg11(pretrained=True).features[:-1]
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.n_anchor = 9
        self.intermediate = nn.Conv2d(512, 512, 3, 1, 1)
        self.cls = nn.Conv2d(512, self.n_anchor * 2, 1, 1, 0)
        self.region = nn.Conv2d(512, self.n_anchor * 4, 1, 1, 0)

        self.n_pre_nms = 12000
        self.nms_thresh = 0.7
        self.n_post_nms = 2000
        anchor_base = np.array([(np.sqrt(area/ratio), np.sqrt(area*ratio))
                                # for ratio in (0.5, 1, 2) for area in (18000, 9000, 2000)], dtype=np.float32)
                                for ratio in (0.5, 1, 2) for area in (18000/9, 9000/9, 2000/9)], dtype=np.float32)
        self.anchor_base = np.concatenate((-anchor_base, anchor_base), axis=1)

    def forward(self, x):
        # (ymin, xmin, ymax, xmax)
        # (y, x, h, w)
        # の順
        # [bg, fg]

        # Backbone, Extractor
        features = self.extractor(x)

        assert x.size(2) % features.size(2) == x.size(3) % features.size(3) == 0
        # RPN
        anchor = self.enumerate_shifted_anchor(x.shape[2:], features.shape[2:])

        h = F.relu(self.intermediate(features))

        # -1 -> features.size(2) * features.size(3) * self.n_anchor
        score = self.cls(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 2))
        offset = self.region(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 4))

        for i, (s, o) in enumerate(zip(score, offset)):
            roi = self.rpn(anchor, o, s, x.size(2), x.size(3))

        # Head
        return x

    def rpn(self, anchor, offset, score, h, w):
        offset = offset.to('cpu')
        score = score.to('cpu')
        # offset = offset.to('cpu')
        # offset, anchor, scoreをcpuに

        roi = torch.empty(anchor.shape)
        center_y = torch.from_numpy((anchor[:, 0] + anchor[:, 2]) / 2)
        center_x = torch.from_numpy((anchor[:, 1] + anchor[:, 3]) / 2)
        height = torch.from_numpy(anchor[:, 2] - anchor[:, 0])
        width = torch.from_numpy(anchor[:, 3] - anchor[:, 1])

        dy, dx, dh, dw = offset.unbind(1)
        dh = 0.5 * torch.exp(dh)
        dw = 0.5 * torch.exp(dw)

        roi[:, 0] = torch.clamp(center_y + height * (dy - dh), 0, h)
        roi[:, 1] = torch.clamp(center_x + width * (dx - dw), 0, w)
        roi[:, 2] = torch.clamp(center_y + height * (dy + dh), 0, h)
        roi[:, 3] = torch.clamp(center_x + width * (dx + dw), 0, w)

        # h = roi.detach().numpy()[:, 2] - roi.detach().numpy()[:, 0]
        # w = roi.detach().numpy()[:, 3] - roi.detach().numpy()[:, 1]
        # keep = np.where((h >= 10) & (w >= 10))[0]
        # score, roi = score[:, keep], roi[keep]

        order = torch.topk(score[:, 1], min(score.size(0), self.n_pre_nms))[1]
        roi = roi[order]
        bbox = roi.detach().numpy()
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        selec = []
        for i, b in enumerate(bbox):
            tl = np.maximum(b[:2], bbox[selec, :2])
            br = np.minimum(b[2:], bbox[selec, 2:])
            area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

            iou = area / (bbox_area[i] + bbox_area[selec] - area)
            if (iou < self.nms_thresh).all():
                selec.append(i)
        roi = roi[selec[:self.n_post_nms]]
        return roi

    def enumerate_shifted_anchor(self, input_size, features_size) -> np.ndarray:
        x_feat_stride = input_size[0] / features_size[0]
        y_feat_stride = input_size[0] / features_size[0]
        shift_x, shift_y = np.meshgrid(np.arange((x_feat_stride-1)/2, input_size[0], x_feat_stride, dtype=np.float32),
                                       np.arange((y_feat_stride-1)/2, input_size[1], y_feat_stride, dtype=np.float32))

        shift = np.stack((shift_y.flatten(), shift_x.flatten(), shift_y.flatten(), shift_x.flatten()), axis=1)
        anchor = (np.expand_dims(shift, 1) + np.expand_dims(self.anchor_base, 0)).reshape(-1, 4)
        return anchor



def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=2,
                        help='1バッチあたり何枚か')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='何epochやるか')
    parser.add_argument('--out', '-o', default='result',
                        help='結果を出力するディレクトリ')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='GPUを使用するか')
    parser.add_argument('--log_interval', '-i', type=int, default=1,
                        help='何iteraionごとに画面に出力するか')
    parser.add_argument('--eval_interval', '-ei', type=int, default=200,
                        help='検証をどの周期で行うか')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from skimage.transform import resize
    dataset = SatelliteImages(
        transform=transforms.Compose((
            # transforms.Resize(256),
            transforms.ToTensor(),)),
        bbox_transform=lambda x: x / 3,
        mask_transform=lambda x: resize(x.transpose((1, 2, 0)), (256, 256), anti_aliasing=True).transpose((2, 0, 1)),
    )

    n = min(64*4, int(len(dataset) * 0.01))
    train_dataset, eval_dataset = random_split(dataset, (len(dataset) - n, n))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, **kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate, **kwargs)

    model = MaskRCNNTrain(MaskRCNN()).to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, eval_loader)


if __name__ == '__main__':
    main()
