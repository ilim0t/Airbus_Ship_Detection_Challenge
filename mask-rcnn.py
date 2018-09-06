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

        # ProposalTargetCreatorの中で
        # self.n_sample, self.pos_ratio, self.pos_iou_thresh, self.neg_iou_thresh_hi, self.neg_iou_thresh_lo
        self.n_sample = 128
        self.pos_ratio = 0.25
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh_hi = 0.5
        self.neg_iou_thresh_lo = 0.0

        # AnchorTargetCreatorの中
        # self.n_sample, self.pos_iou_thresh, self.neg_iou_thresh, self.pos_ratio
        n_sample = 256
        pos_iou_thresh = 0.7
        neg_iou_thresh = 0.3
        pos_ratio = 0.5

    def forward(self, x: torch.Tensor, gt_bboxes: Tuple[np.ndarray], masks: Tuple[np.ndarray]):
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
        scores = self.mask_rcnn.cls(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 2))
        offsets = self.mask_rcnn.region(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 4))

        for i, (feature, score, offset, gt_bbox, gt_mask) in enumerate(zip(features, scores, offsets, gt_bboxes, masks)):
            # nms
            # scoreのメモリに上書きされる(無駄で冗長な処理)かも
            offset, score, roi = self.mask_rcnn.rpn(anchor, offset, score, x.size(2), x.size(3))

            sampled_roi, nearest_gt_bbox, gt_corresponded_label, sampled_loc = self.proposal_target_creator(roi, gt_bbox)
            head_cls_loc, head_score, mask = self.mask_rcnn.head(feature, sampled_roi.detach().numpy(), x.shape[2:])
            gt_loc, gt_corresponded_label2 = self.anchor_target_creator(anchor, gt_bbox, x.shape[2:])

            rpn_loc_loss = self.loc_loss(offset, gt_loc, gt_corresponded_label2)
            rpn_cls_loss = F.cross_entropy(score, gt_corresponded_label2)
            roi_loc_loss = self.loc_loss(head_cls_loc[np.arange(len(head_cls_loc)), gt_corresponded_label], sampled_loc, gt_corresponded_label)
            roi_cls_loss = F.cross_entropy(head_score, gt_corresponded_label)

            loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

    def calc_iou(self, bbox: np.ndarray, gt_bbox: np.ndarray) -> np.ndarray:
        tl = np.maximum(bbox[:, None, :2], gt_bbox[:, :2])  # [..., None, ...] は np.expand_dim(input, n) と等価
        br = np.minimum(bbox[:, None, 2:], gt_bbox[:, 2:])

        cap = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        gt_area = np.prod(gt_bbox[:, 2:] - gt_bbox[:, :2], axis=1)
        iou = cap / (area[:, None] + gt_area - cap)
        return iou

    def bbox2loc(self, bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        height = bbox[:, 2] - bbox[:, 0]
        width = bbox[:, 3] - bbox[:, 1]

        dy = (gt_bbox[:, 2] + gt_bbox[:, 0] - bbox[:, 2] - bbox[:, 0]) / 2 / height
        dx = (gt_bbox[:, 3] + gt_bbox[:, 2] - bbox[:, 3] - bbox[:, 2]) / 2 / width
        dh = torch.log(height - gt_bbox[:, 2] + gt_bbox[:, 0])
        dw = torch.log(width - gt_bbox[:, 3] + gt_bbox[:, 1])

        loc = torch.stack((dy, dx, dh, dw)).transpose(0, 1)
        return loc

    def np_bbox2loc(self, bbox: np.ndarray, gt_bbox: np.ndarray) -> np.ndarray:
        height = bbox[:, 2] - bbox[:, 0]
        width = bbox[:, 3] - bbox[:, 1]

        dy = (gt_bbox[:, 2] + gt_bbox[:, 0] - bbox[:, 2] - bbox[:, 0]) / 2 / height
        dx = (gt_bbox[:, 3] + gt_bbox[:, 2] - bbox[:, 3] - bbox[:, 2]) / 2 / width
        dh = np.log(height - gt_bbox[:, 2] + gt_bbox[:, 0])
        dw = np.log(width - gt_bbox[:, 3] + gt_bbox[:, 1])

        loc = np.stack((dy, dx, dh, dw)).transpose(0, 1)
        return loc

    def proposal_target_creator(self, bbox: torch.Tensor, gt_bbox: np.ndarray) \
            -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        # proposal_target_creator相当
        # roi = np.concatenate((roi, bbox))
        iou = self.calc_iou(bbox.detach().numpy(), gt_bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(np.round(self.n_sample * self.pos_ratio), pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = int(min(self.n_sample - pos_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_label = np.ones_like(keep_index)
        gt_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        bbox = bbox[keep_index]

        gt_bbox = torch.from_numpy(gt_bbox[gt_assignment[keep_index]])
        loc = self.bbox2loc(bbox, gt_bbox)
        loc = loc / torch.Tensor([0.1, 0.1, 0.2, 0.2])
        return bbox, gt_bbox, gt_label, loc

    def anchor_target_creator(self, anchor: np.ndarray, bbox: np.ndarray, size: Tuple[int, int]) \
            -> Tuple[np.ndarray, np.ndarray]:

        n_sample = 256
        pos_iou_thresh = 0.7
        neg_iou_thresh = 0.3
        pos_ratio = 0.5

        n_anchor = len(anchor)

        keep = np.where((0 <= anchor[:, 0]) & (0 <= anchor[:, 1]) & (anchor[:, 2] <= size[0]) & (anchor[:, 3] <= size[1]))[0]
        anchor = anchor[keep]

        label = np.empty((len(keep), ), dtype=np.int32)
        label.fill(-1)

        ious = self.calc_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(anchor)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(pos_ratio * n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        loc = self.np_bbox2loc(anchor, bbox[argmax_ious])

        label = self._unmap(label, n_anchor, keep, fill=-1)
        loc = self._unmap(loc, n_anchor, keep, fill=0)
        return loc, label

    def _unmap(self, data, count, index, fill=0):
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=data.dtype)
            ret.fill(fill)
            ret[index] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[index, :] = data
        return ret


class MaskRCNN(nn.Module):
    def __init__(self, class_num=1):
        super(MaskRCNN, self).__init__()
        self.extractor = models.vgg11(pretrained=True).features[:-1]
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.n_pre_nms = 12000
        self.nms_thresh = 0.7
        self.n_post_nms = 2000
        anchor_base = np.array([(np.sqrt(area/ratio), np.sqrt(area*ratio))
                                # for ratio in (0.5, 1, 2) for area in (18000, 9000, 2000)], dtype=np.float32)
                                for ratio in (0.5, 1, 2) for area in (18000/9, 9000/9, 2000/9, 30)], dtype=np.float32)
        self.n_anchor = len(anchor_base)
        self.anchor_base = np.concatenate((-anchor_base, anchor_base), axis=1)

        # RPN
        self.intermediate = nn.Conv2d(512, 512, 3, 1, 1)
        self.cls = nn.Conv2d(512, self.n_anchor * 2, 1, 1, 0)
        self.region = nn.Conv2d(512, self.n_anchor * 4, 1, 1, 0)

        # HEAD
        self.fc6 = nn.Linear(512*7*7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.cls_loc = nn.Linear(4096, 2 * 4)
        self.score = nn.Linear(4096, 2)

        self.conv1_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(256, 2, 3, 1, 1)

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
            bbox, score_ = self.rpn(anchor, o, s, x.size(2), x.size(3))

        # Head
        return x

    def rpn(self, anchor: np.ndarray, offset: torch.Tensor, score: torch.Tensor, h: int, w: int)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        bbox = self.offset2bbox(offset, anchor, h, w)

        # h = roi.detach().numpy()[:, 2] - roi.detach().numpy()[:, 0]
        # w = roi.detach().numpy()[:, 3] - roi.detach().numpy()[:, 1]
        # keep = np.where((h >= 10) & (w >= 10))[0]
        # score, roi = score[:, keep], roi[keep]

        bbox, score, offset = self.nms(bbox, score, offset)
        return offset, score, bbox

    def head(self, x: torch.Tensor, sampled_roi: np.ndarray, size: Tuple[int, int]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_roi = np.concatenate((sampled_roi, np.zeros((sampled_roi.shape[0], 1), dtype=np.float32)), axis=1)
        pool = self.pooling(x.unsqueeze(0), sampled_roi, 7, size)
        h = F.relu(self.fc6(pool.flatten(1)))
        h = F.relu(self.fc7(h))
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)

        h = F.relu(self.conv1_1(pool))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.deconv(h))
        h = F.relu(self.conv2_1(h))
        roi_mask = self.conv2_2(h)
        return roi_cls_locs, roi_scores, roi_mask

    def pooling(self, x: torch.Tensor, bbox: np.ndarray, pool_size: int=7, origin_size=(256, 256)) -> torch.Tensor:
        height = x.size(2)
        width = x.size(3)

        top = torch.from_numpy(bbox[:, 1::4] / origin_size[0] * height)
        left = torch.from_numpy(bbox[:, 2::4] / origin_size[1] * width)
        bottom = torch.from_numpy(bbox[:, 3::4] / origin_size[0] * height)
        right = torch.from_numpy(bbox[:, 4::4] / origin_size[1] * width)

        # affine theta
        zero = torch.zeros(bbox.shape[0], 1)
        theta = torch.cat([
            (bottom - top) / (width - 1),
            zero,
            (top + bottom - width + 1) / (width - 1),
            zero,
            (right - left) / (height - 1),
            (left + right - height + 1) / (height - 1)
        ], 1).view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((bbox.shape[0], 1, pool_size, pool_size)))
        crops = F.grid_sample(x.expand(bbox.shape[0], *x.shape[1:]), grid)
        return crops

    def offset2bbox(self, offset: torch.Tensor, anchor: np.ndarray, h: int, w: int) -> torch.Tensor:
        bbox = torch.empty(anchor.shape)
        center_y = torch.from_numpy((anchor[:, 0] + anchor[:, 2]) / 2)
        center_x = torch.from_numpy((anchor[:, 1] + anchor[:, 3]) / 2)
        height = torch.from_numpy(anchor[:, 2] - anchor[:, 0])
        width = torch.from_numpy(anchor[:, 3] - anchor[:, 1])

        dy, dx, dh, dw = offset.unbind(1)
        dh = 0.5 * torch.exp(dh)
        dw = 0.5 * torch.exp(dw)

        bbox[:, 0] = torch.clamp(center_y + height * (dy - dh), 0, h)
        bbox[:, 1] = torch.clamp(center_x + width * (dx - dw), 0, w)
        bbox[:, 2] = torch.clamp(center_y + height * (dy + dh), 0, h)
        bbox[:, 3] = torch.clamp(center_x + width * (dx + dw), 0, w)
        return bbox

    def nms(self, bbox: torch.Tensor, score: torch.Tensor, offset: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.topk(score[:, 1], min(score.size(0), self.n_pre_nms))[1]
        bbox = bbox[order]
        score = score[order]
        np_bbox = bbox.detach().numpy()
        bbox_area = np.prod(np_bbox[:, 2:] - np_bbox[:, :2], axis=1)
        selec = []
        for i, b in enumerate(np_bbox):
            tl = np.maximum(b[:2], np_bbox[selec, :2])
            br = np.minimum(b[2:], np_bbox[selec, 2:])
            cap = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

            iou = cap / (bbox_area[i] + bbox_area[selec] - cap)
            if (iou < self.nms_thresh).all():
                selec.append(i)

        bbox = bbox[selec[:self.n_post_nms]]
        score = score[selec[:self.n_post_nms]]
        offset = offset[selec[:self.n_post_nms]]
        return bbox, score, offset

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
    torch.manual_seed(2)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    from skimage.transform import resize
    dataset = SatelliteImages(
        transform=transforms.Compose((
            transforms.Resize(256),
            transforms.ToTensor())),
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
