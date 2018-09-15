#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision import utils

import argparse
import json
import numpy as np
from typing import Tuple
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import itertools

from dataset import SatelliteImages, MaskOnly
from util import decode, show
from repoter import Repoter


def train(model, device, train_loader, optimizer, reporter, step):
    data, bboxs, masks = next(train_loader)
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    loss, rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, mask_loss, bbox, mask, (gt_label, predict_cls) = model(data, bboxs, masks)
    loss.backward()
    optimizer.step()

    reporter.report({"train/loss": loss.item()})
    reporter.writer.add_scalar("train/loss", loss.item(), step)
    reporter.writer.add_scalar("train/rpn_cls_loss", rpn_cls_loss.item(), step)
    reporter.writer.add_pr_curve("train/rpn_cls", gt_label[gt_label >= 0], F.softmax(predict_cls)[gt_label >= 0, 1], step)
    # gt_label[gt_label >= 0].numpy(), F.softmax(predict_cls)[gt_label >= 0, 1].numpy()
    if rpn_loc_loss is not None:
        reporter.writer.add_scalar("train/rpn_loc_loss", rpn_loc_loss.item(), step)
        reporter.writer.add_scalar("train/roi_loc_loss", roi_loc_loss.item(), step)
        reporter.writer.add_scalar("train/roi_cls_loss", roi_cls_loss.item(), step)
        if bbox.numel():
            bbox[:, 0] = torch.clamp(bbox[:, 0], 0, data.size(2))
            bbox[:, 1] = torch.clamp(bbox[:, 1], 0, data.size(3))
            bbox[:, 2] = torch.clamp(bbox[:, 2], 0, data.size(2))
            bbox[:, 3] = torch.clamp(bbox[:, 3], 0, data.size(3))
            reporter.writer.add_scalar("train/mask_loss", mask_loss.item(), step)
            reporter.writer.add_image_with_boxes('train/bbox', data[0], torch.cat((torch.zeros(bbox.size(0), 1), bbox), dim=1), step)
        else:
            ...
    # reporter.writer.add_histogram("intermediate.W", model.mask_rcnn.intermediate.weight, step)
    if bboxs[0] is not None:
        reporter.writer.add_image_with_boxes('train/gt_bbox', data[0], torch.cat((torch.zeros(bboxs[0].shape[0], 1), torch.from_numpy(bboxs[0])), dim=1), step)


def eval(model, device, eval_loader, reporter, step):
    model.eval()
    eval_loss = 0
    accuracy = []
    # cum_output = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += F.binary_cross_entropy(output, target)

            predict = output > 0.5
            target = target > 0.5

            accuracy.append((predict == target).float())

    eval_loss /= len(eval_loader)
    accuracy = torch.cat(accuracy).mean()

    # cum_output = torch.cat(cum_output).view(-1)

    reporter.report({"eval/loss": eval_loss.item(),
                     "eval/accuracy": accuracy.item()})
    reporter.writer.add_scalar("eval/loss", eval_loss.item(), step)
    reporter.writer.add_scalar("eval/accuracy", accuracy.item(), step)
    # reporter.writer.add_pr_curve("eval", ground_true, cum_output, step)


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

        # self._n = 0
        self._n = 200

    def forward(self, x: torch.Tensor, gt_bboxes: Tuple[np.ndarray], gt_masks: Tuple[np.ndarray]):
        # (ymin, xmin, ymax, xmax)
        # (y, x, h, w)
        # の順
        # [bg, fg]
        assert x.size(0) == 1, "現在 minibatch == 1 のみ対応しています"

        # Backbone, Extractor
        features = self.mask_rcnn.extractor(x)
        features.requires_grad = True
        assert x.size(2) % features.size(2) == x.size(3) % features.size(3) == 0

        # RPN
        anchor = self.mask_rcnn.enumerate_shifted_anchor(x.shape[2:], features.shape[2:])
        h = F.relu(self.mask_rcnn.intermediate(features))
        scores = self.mask_rcnn.cls(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 2))
        offsets = self.mask_rcnn.region(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 4))

        for i, (feature, score, offset, gt_bbox, gt_mask) in enumerate(zip(features, scores, offsets, gt_bboxes, gt_masks)):
            gt_bbox = torch.from_numpy(gt_bbox) if gt_bbox is not None else None
            gt_mask = torch.from_numpy(gt_mask) if gt_mask is not None else None

            self._n += 1

            roi = self.mask_rcnn.rpn(anchor, offset, score.detach(), x.shape[2:4])
            if self._n > 200 and gt_bbox is not None:
                # print("gt_bbox area: " + ",".join([str(num.numpy()) for num in (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])]))
                sampled_roi, sampled_loc, gt_roi_label, gt_roi_mask = self.proposal_target_creator(roi, gt_bbox, gt_mask, x.shape[2:])
                head_cls_loc, head_score, mask = self.mask_rcnn.head(feature, sampled_roi.detach(), x.shape[2:])
            gt_loc_with_anchor, gt_rpn_label = self.anchor_target_creator(anchor, gt_bbox, x.shape[2:])
            if self._n > 200 and gt_bbox is not None:
                rpn_loc_loss = self.loc_loss(offset, gt_loc_with_anchor, gt_rpn_label)
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label, ignore_index=-1)
            if self._n > 200 and gt_bbox is not None:
                roi_loc_loss = self.loc_loss(head_cls_loc[np.arange(len(head_cls_loc)), gt_roi_label],
                                             sampled_loc, gt_roi_label)  # inf
                roi_cls_loss = F.cross_entropy(head_score, gt_roi_label, ignore_index=-1)
                mask_loss = F.cross_entropy(mask, gt_roi_mask) if torch.max(gt_roi_mask) == 1 else torch.zeros(1)

            if self._n > 200 and gt_bbox is not None:
                loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + mask_loss
                argmax = torch.argmax(head_score, 1)
                return loss, rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, mask_loss,\
                       head_cls_loc[argmax == 1, 1, :], mask[argmax == 1], (gt_rpn_label, score)
                       # head_cls_loc[argmax == 1, 1, :], mask[argmax == 1]
            else:
                loss = rpn_cls_loss
                return loss, None, rpn_cls_loss, None, None, None, None, None, (gt_rpn_label, score)

    def calc_iou(self, bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        assert not bbox.requires_grad
        assert gt_bbox is None or not gt_bbox.requires_grad
        if gt_bbox is None:
            return torch.zeros(bbox.size(0), 1)
        tl = torch.max(bbox[:, None, :2], gt_bbox[:, :2])  # [..., None, ...] は np.expand_dim(input, n) と等価
        br = torch.min(bbox[:, None, 2:], gt_bbox[:, 2:])

        cap = torch.where((tl < br).all(2), (br - tl).prod(2), torch.zeros(1))
        area = (bbox[:, 2:] - bbox[:, :2]).prod(1)
        gt_area = (gt_bbox[:, 2:] - gt_bbox[:, :2]).prod(1)
        iou = cap / (area.unsqueeze(1) + gt_area - cap)
        return iou

    def bbox2loc(self, bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
        assert not bbox.requires_grad
        assert not gt_bbox.requires_grad

        height = torch.clamp(bbox[:, 2] - bbox[:, 0], min=1e-2)  # bbox = []
        width = torch.clamp(bbox[:, 3] - bbox[:, 1], min=1e-2)

        dy = (gt_bbox[:, 2] + gt_bbox[:, 0] - bbox[:, 2] - bbox[:, 0]) / 2 / height
        dx = (gt_bbox[:, 3] + gt_bbox[:, 2] - bbox[:, 3] - bbox[:, 2]) / 2 / width
        dh = torch.log((gt_bbox[:, 2] - gt_bbox[:, 0]) / height)
        dw = torch.log((gt_bbox[:, 3] - gt_bbox[:, 1]) / width)

        loc = torch.stack((dy, dx, dh, dw), dim=1)
        return loc

    def loc_loss(self, loc: torch.Tensor, gt_loc: torch.Tensor, gt_label: torch.Tensor, sigma: float=1.) \
            -> torch.Tensor:
        assert loc.requires_grad
        assert not gt_loc.requires_grad
        assert not gt_label.requires_grad

        sigma2 = sigma ** 2
        diff = (loc - gt_loc)
        abs_diff = diff.abs()
        loss = torch.where(gt_label >= 0, torch.where(
            abs_diff < 1 / sigma2, diff.pow(2) * sigma2 / 2, abs_diff - 0.5 / sigma2).sum(1), torch.zeros(1)).sum() / \
            (gt_label >= 0).sum().float()
        return loss

    def proposal_target_creator(self, bbox: torch.Tensor, gt_bbox: torch.Tensor, gt_mask: torch.Tensor, size: Tuple[int, int]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert bbox.requires_grad
        assert gt_bbox is None or not gt_bbox.requires_grad
        assert gt_mask is None or not gt_mask.requires_grad
        # bbox = torch.cat((bbox, gt_bbox))

        iou = self.calc_iou(bbox.detach(), gt_bbox)
        max_iou, gt_assignment = iou.max(dim=1)
        # gt_max_ious, gt_argmax_ious = iou.max(0)
        # gt_argmax_ious = np.where(iou == gt_max_ious)[0]

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # pos_index = np.where((max_iou >= self.pos_iou_thresh).numpy() | np.any((iou == gt_max_ious).numpy(), 1))[0]
        pos_roi_per_this_image = int(min(np.round(self.n_sample * self.pos_ratio), pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = int(min(self.n_sample - pos_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        keep = np.append(pos_index, neg_index)
        gt_label = torch.cat((torch.ones(pos_roi_per_this_image, dtype=torch.long),
                              torch.zeros(neg_roi_per_this_image, dtype=torch.long)))

        gt_assignment = gt_assignment[keep]
        bbox = bbox[keep]
        gt_bbox = gt_bbox[gt_assignment]
        gt_mask = gt_mask[gt_assignment]

        loc = self.bbox2loc(bbox.detach(), gt_bbox[gt_assignment])
        loc = loc / torch.Tensor([0.1, 0.1, 0.2, 0.2])

        sample = torch.cat((torch.zeros(bbox.size(0)).unsqueeze(1), bbox.detach()), dim=1)
        gt_pooled_mask = self.mask_rcnn.pooling(gt_mask.unsqueeze(1), sample, 7 * 2, size).squeeze(1).long()  # 256に縮小しているので小数がある longにするとそれが失われる
        """
        show(gt_mask[0]), show(gt_mask[1])
        gt_pooled_mask = self.mask_rcnn.pooling(gt_mask.unsqueeze(1), torch.cat((torch.zeros(gt_bbox.size(0)).unsqueeze(1), gt_bbox), dim=1), 7, size).squeeze(1)
        show(gt_pooled_mask[0]), show(gt_pooled_mask[1])
        """
        return bbox, loc, gt_label, gt_pooled_mask

    def anchor_target_creator(self, anchor: torch.Tensor, bbox: torch.Tensor, size: Tuple[int, int]) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert not anchor.requires_grad
        # assert not bbox.requires_grad
        # anchor = torch.cat((anchor, bbox))

        n_sample = 256
        pos_iou_thresh = 0.7
        neg_iou_thresh = 0.3
        pos_ratio = 0.5

        n_anchor = len(anchor)
        np_anchor = anchor.numpy()

        keep = np.where((0 <= np_anchor[:, 0]) & (0 <= np_anchor[:, 1]) &
                        (np_anchor[:, 2] <= size[0]) & (np_anchor[:, 3] <= size[1]))[0]
        anchor = anchor[keep]

        label = - torch.ones(keep.shape, dtype=torch.long)

        ious = self.calc_iou(anchor, bbox)
        max_ious, argmax_ious = ious.max(1)
        gt_max_ious, gt_argmax_ious = ious.max(0)
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        label[max_ious < neg_iou_thresh] = 0
        if gt_max_ious.max().item() != 0:
            label[gt_argmax_ious] = 1
        label[max_ious >= pos_iou_thresh] = 1

        n_pos = int(pos_ratio * n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = n_sample - torch.sum(label == 1).item()
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        loc = self.bbox2loc(anchor, bbox[argmax_ious]) if bbox is not None else None

        label = self._unmap(label, n_anchor, keep, fill=-1)
        loc = self._unmap(loc, n_anchor, keep, fill=0) if bbox is not None else None
        return loc, label

    def _unmap(self, data: torch.Tensor, count: int, index: np.ndarray, fill: float=0):
        if len(data.shape) == 1:
            ret = torch.empty(count, dtype=data.dtype).fill_(fill)
            ret[index] = data
        else:
            ret = torch.empty((count,) + data.shape[1:], dtype=data.dtype).fill_(fill)
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
        anchor_base = np.array([(np.sqrt(area / ratio) / 2, np.sqrt(area * ratio) / 2)
                                for ratio in (0.5, 1, 2) for area in (80, 400, 1000)], dtype=np.float32)

        self.n_anchor = len(anchor_base)
        self.anchor_base = torch.from_numpy(np.concatenate((-anchor_base, anchor_base), axis=1))

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
        features = self.mask_rcnn.extractor(x)
        assert x.size(2) % features.size(2) == x.size(3) % features.size(3) == 0

        # RPN
        anchor = self.mask_rcnn.enumerate_shifted_anchor(x.shape[2:], features.shape[2:])
        h = F.relu(self.mask_rcnn.intermediate(features))
        scores = self.mask_rcnn.cls(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 2))
        offsets = self.mask_rcnn.region(h).permute((0, 2, 3, 1)).contiguous().view((x.size(0), -1, 4))

        for i, (feature, score, offset) in enumerate(zip(features, scores, offsets)):
            roi = self.mask_rcnn.rpn(anchor, offset, score.detach(), x.shape[2:4])
            head_cls_loc, head_score, mask = self.mask_rcnn.head(feature, roi, x.shape[2:])
            raise SystemExit
        # return mask

    def rpn(self, anchor: torch.Tensor, offset: torch.Tensor, score: torch.Tensor, size: Tuple[int, int]) \
            -> torch.Tensor:
        assert not anchor.requires_grad
        assert offset.requires_grad
        assert not score.requires_grad

        bbox = self.offset2bbox(offset, anchor, size)

        # h = roi.detach().numpy()[:, 2] - roi.detach().numpy()[:, 0]
        # w = roi.detach().numpy()[:, 3] - roi.detach().numpy()[:, 1]
        # keep = np.where((h >= 10) & (w >= 10))[0]
        # score, roi = score[:, keep], roi[keep]

        bbox = self.nms(bbox, score)
        return bbox

    def head(self, x: torch.Tensor, sampled_roi: torch.Tensor, size: Tuple[int, int]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.requires_grad
        assert not sampled_roi.requires_grad

        sampled_roi = torch.cat((torch.zeros(sampled_roi.size(0)).unsqueeze(1), sampled_roi), dim=1)
        pool = self.pooling(x.expand(128, -1, -1, -1), sampled_roi, 7, size)

        h = F.relu(self.fc6(pool.flatten(1)))
        h = F.relu(self.fc7(h))
        roi_cls_locs = self.cls_loc(h).view(-1, 2, 4)
        roi_scores = self.score(h)

        h = F.relu(self.conv1_1(pool))
        h = F.relu(self.conv1_2(h))
        h = F.relu(self.deconv(h))
        h = F.relu(self.conv2_1(h))
        roi_mask = self.conv2_2(h)
        return roi_cls_locs, roi_scores, roi_mask

    def pooling(self, x: torch.Tensor, bbox: torch.Tensor, pool_size: int=7, origin_size=(256, 256)) -> torch.Tensor:
        # assert x.requires_grad
        assert not bbox.requires_grad

        height = x.size(2)
        width = x.size(3)

        top = bbox[:, 1::4] / origin_size[0] * height
        left = bbox[:, 2::4] / origin_size[1] * width
        bottom = bbox[:, 3::4] / origin_size[0] * height
        right = bbox[:, 4::4] / origin_size[1] * width

        zero = torch.zeros(bbox.size(0)).unsqueeze(1)
        theta = torch.cat([
            (bottom - top) / (width - 1),
            zero,
            (top + bottom - width + 1) / (width - 1),
            zero,
            (right - left) / (height - 1),
            (left + right - height + 1) / (height - 1)
        ], 1).view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((bbox.shape[0], 1, pool_size, pool_size)))
        crops = F.grid_sample(x, grid)
        return crops

    def offset2bbox(self, offset: torch.Tensor, anchor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:  # checked
        assert offset.requires_grad
        assert not anchor.requires_grad

        center_y = (anchor[:, 0] + anchor[:, 2]) / 2
        center_x = (anchor[:, 1] + anchor[:, 3]) / 2
        height = anchor[:, 2] - anchor[:, 0]
        width = anchor[:, 3] - anchor[:, 1]

        dy, dx, dh, dw = offset.unbind(1)
        dh = 0.5 * torch.exp(dh)
        dw = 0.5 * torch.exp(dw)

        bbox = torch.stack((torch.clamp(center_y + height * (dy - dh), 0, size[0]),
                            torch.clamp(center_x + width * (dx - dw), 0, size[1]),
                            torch.clamp(center_y + height * (dy + dh), 0, size[0]),
                            torch.clamp(center_x + width * (dx + dw), 0, size[1])), dim=1)
        return bbox

    def nms(self, bbox: torch.Tensor, score: torch.Tensor) -> torch.Tensor:  # checked
        assert bbox.requires_grad
        assert not score.requires_grad

        order = torch.topk(score[:, 1], min(score.size(0), self.n_pre_nms))[1]
        bbox = bbox[order]
        dt_bbox = bbox.detach()

        area = (dt_bbox[:, 2:] - dt_bbox[:, :2]).prod(1)
        selec = []
        for i, b in enumerate(dt_bbox):
            if not selec:
                selec.append(i)
            tl = torch.max(b[:2], dt_bbox[selec, :2])
            br = torch.min(b[2:], dt_bbox[selec, 2:])

            cap = torch.where((tl < br).all(1), (br - tl).prod(1), torch.zeros(1))

            iou = cap / (area[i] + area[selec] - cap)  # squeezeは?
            if (iou < self.nms_thresh).all():
                selec.append(i)

        if self.n_post_nms < len(selec):
            selec = selec[:self.n_post_nms]

        bbox = bbox[selec]
        return bbox

    def enumerate_shifted_anchor(self, input_size: Tuple[int, int], features_size: Tuple[int, int]) -> torch.Tensor:
        x_feat_stride = input_size[0] / features_size[0]
        y_feat_stride = input_size[0] / features_size[0]
        shift_y, shift_x = torch.meshgrid(
            (torch.arange(x_feat_stride / 2, input_size[0], x_feat_stride),
             torch.arange(y_feat_stride / 2, input_size[1], y_feat_stride)))

        shift = torch.stack((shift_y.flatten(), shift_x.flatten(), shift_y.flatten(), shift_x.flatten()), dim=1)
        anchor = (shift.unsqueeze(1) + self.anchor_base.unsqueeze(0)).view(-1, 4)
        return anchor


def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
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
    dataset = MaskOnly(
        transform=transforms.Compose((
            transforms.Resize(256),
            transforms.ToTensor())),
        bbox_transform=lambda x: x / 3,
        mask_transform=lambda x: resize(x.transpose((1, 2, 0)), (256, 256), mode='reflect', anti_aliasing=True).
            transpose((2, 0, 1)).astype(np.float32),
    )

    n = min(64*4, int(len(dataset) * 0.01))
    train_dataset, eval_dataset = random_split(dataset, (len(dataset) - n, n))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, **kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate, **kwargs)

    model = MaskRCNNTrain(MaskRCNN()).to(device)
    optimizer = optim.Adam(model.parameters())

    writer = SummaryWriter(
        os.path.join(args.out, datetime.now().strftime('%m-%d_%H:%M:%S_bs-{}'.format(args.batch_size))))

    reporter = Repoter(
        ['epoch', 'iteration', 'train/loss', 'train/accuracy', 'eval/loss', 'eval/accuracy', 'elapsed_time'],
        writer, trigger=(args.log_interval, 'iteration'), all_epoch=args.epochs, iter_per_epoch=len(train_loader))

    # torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(int)

    for epoch_idx in range(1, args.epochs + 1) if args.epochs > 0 else itertools.count(1):
        train_iter = iter(train_loader)
        for iteration_idx in range(1, len(train_loader) + 1):
            step = iteration_idx + (epoch_idx - 1) * len(train_loader)

            with reporter.scope(epoch_idx - 1, step):
                train(model, device, train_iter, optimizer, reporter, step)
                # if step % args.eval_interval == 0:
                #     return
                #     eval(model, device, eval_loader, reporter, step)
        torch.save(model.state_dict(), "snapshot")


if __name__ == '__main__':
    main()
