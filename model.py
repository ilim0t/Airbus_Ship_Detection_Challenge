#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3)

        self.fc = nn.Linear(16, 1)

    def forward(self, x):  # => 3, 384
        x = F.max_pool2d(self.conv1(x), 2)  # => 32, 95
        x = F.max_pool2d(self.conv2(F.relu(x)), 2)  # => 32, 46
        x = F.dropout(x, 0.3)
        x = self.bn2(x)
        x = F.max_pool2d(self.conv3(F.relu(x)), 2)  # => 32, 22
        x = F.max_pool2d(self.conv4(F.relu(x)), 2)  # => 32, 10
        x = F.dropout(x, 0.3)
        x = self.bn4(x)
        x = F.max_pool2d(self.conv5(F.relu(x)), 2)  # => 32, 4
        x = F.max_pool2d(self.conv6(F.relu(x)), 2)  # => 32, 1

        x = self.fc(x.view(x.size(0), -1))
        return torch.sigmoid(x)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = UnetBlock(3, 32)
        self.down1 = UnetBlock(32, 64)
        self.down2 = UnetBlock(64, 128)
        self.down3 = UnetBlock(128, 256)

        self.conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.upsamp3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = UnetBlock(512, 256)

        self.upsamp2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = UnetBlock(256, 128)

        self.upsamp1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = UnetBlock(128, 64)

        self.upsamp0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=3)
        self.outconv = nn.Conv2d(64+32, 1, kernel_size=1)

    def forward(self, x):  # => 3, 768 (=2^8*3)
        x0 = self.down0(x)  # => 32, 512
        x1 = self.down1(F.max_pool2d(x0, 3))  # => 64, 256
        x2 = self.down2(F.max_pool2d(x1, 2))  # => 128, 128
        x3 = self.down3(F.max_pool2d(x2, 2))  # => 256, 64

        x4 = self.conv(F.max_pool2d(x3, 2))  # => 512, 32

        x3 = self.up3(torch.cat((x3, self.upsamp3(x4)), dim=1))  # => 256, 64
        x2 = self.up1(torch.cat((x2, self.upsamp2(x3)), dim=1))  # => 128, 128
        x1 = self.up1(torch.cat((x1, self.upsamp1(x2)), dim=1))  # => 64, 256

        x0 = self.outconv(torch.cat((x0, self.upsamp0(x1)), dim=1))  # => 1, 768
        x0 = torch.sigmoid(x0)
        return torch.squeeze(x0, dim=1)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        return x


class Unet2(Unet):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = UnetBlock(3, 32)
        self.down1 = UnetBlock(32, 64)
        self.down2 = UnetBlock(64, 128)

        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.upsamp2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = UnetBlock(256, 128)

        self.upsamp1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = UnetBlock(128, 64)

        self.upsamp0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.outconv = nn.Conv2d(64+32, 1, kernel_size=1)
        self.resize = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):  # => 3, 256
        x0 = self.down0(x)  # => 32, 256
        x1 = self.down1(F.max_pool2d(x0, 2))  # => 64, 128
        x2 = self.down2(F.max_pool2d(x1, 2))  # => 128, 64

        x3 = self.conv(F.max_pool2d(x2, 2))  # => 256, 32

        x2 = self.up2(torch.cat((x2, self.upsamp2(x3)), dim=1))  # => 128, 128
        x1 = self.up1(torch.cat((x1, self.upsamp1(x2)), dim=1))  # => 64, 256

        x0 = self.outconv(torch.cat((x0, self.upsamp0(x1)), dim=1))  # => 1, 768
        x0 = self.resize(x0)
        x0 = torch.sigmoid(x0)
        return torch.squeeze(x0, dim=1)


class Shave(nn.Module):
    def __init__(self):
        super(Shave, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

    def forward(self, x):  # -> (5, 4, 32, 32)
        x = self.conv1(x)  # 3
        x = self.conv2(x)  # 5
        x = F.max_pool2d(x, 2)  # 7
        x = self.conv3(x)  # 9
        x = self.conv4(x)  # 11
        x = F.max_pool2d(x, 2)  # 13

        width = x.size(3)
        x = x.view((*x.shape[:-2], -1))  # -> (5, 256, 1024)
        x.topk(int(x.size(2)* 0.4))  # -> (5, 256, n) * 2

        # 128, 128 (/6^2)の格子で 畳み込む
        return x


class MaskRCNN(nn.Module):
    def __init__(self, class_num=1):
        super(MaskRCNN, self).__init__()
        self.extractor = models.vgg11(pretrained=True).features[:-1]
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Backbone, Extractor -> VGG: /32^2, Resnet: /32^2
        features = self.extractor(x)

        # RPN

        # Head
        return x


#
# import numpy as np
#
# import chainer
# from chainer.backends import cuda
# import chainer.functions as F
#
# from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
#     AnchorTargetCreator
# from chainercv.links.model.faster_rcnn.utils.proposal_target_creator import\
#     ProposalTargetCreator
#
#
# class FasterRCNNTrainChain(chainer.Chain):
#     def __init__(self, faster_rcnn, rpn_sigma=3., roi_sigma=1.,
#                  anchor_target_creator=AnchorTargetCreator(),
#                  proposal_target_creator=ProposalTargetCreator()):
#         super(FasterRCNNTrainChain, self).__init__()
#         with self.init_scope():
#             self.faster_rcnn = faster_rcnn
#         self.rpn_sigma = rpn_sigma
#         self.roi_sigma = roi_sigma
#
#         self.anchor_target_creator = anchor_target_creator
#         self.proposal_target_creator = proposal_target_creator
#
#         self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
#         self.loc_normalize_std = faster_rcnn.loc_normalize_std
#
#     def __call__(self, imgs, bboxes, labels, scale):
#         """Forward Faster R-CNN and calculate losses.
#
#         Here are notations used.
#
#         * :math:`N` is the batch size.
#         * :math:`R` is the number of bounding boxes per image.
#
#         Currently, only :math:`N=1` is supported.
#
#         Args:
#             imgs (~chainer.Variable): A variable with a batch of images.
#             bboxes (~chainer.Variable): A batch of bounding boxes.
#                 Its shape is :math:`(N, R, 4)`.
#             labels (~chainer.Variable): A batch of labels.
#                 Its shape is :math:`(N, R)`. The background is excluded from
#                 the definition, which means that the range of the value
#                 is :math:`[0, L - 1]`. :math:`L` is the number of foreground
#                 classes.
#             scale (float or ~chainer.Variable): Amount of scaling applied to
#                 the raw image during preprocessing.
#
#         Returns:
#             chainer.Variable:
#             Scalar loss variable.
#             This is the sum of losses for Region Proposal Network and
#             the head module.
#
#         """
#         if isinstance(bboxes, chainer.Variable):
#             bboxes = bboxes.array
#         if isinstance(labels, chainer.Variable):
#             labels = labels.array
#         if isinstance(scale, chainer.Variable):
#             scale = scale.array
#         scale = np.asscalar(cuda.to_cpu(scale))
#         n = bboxes.shape[0]
#         if n != 1:
#             raise ValueError('Currently only batch size 1 is supported.')
#
#         _, _, H, W = imgs.shape
#         img_size = (H, W)
#
#         features = self.faster_rcnn.extractor(imgs)
#         rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
#             features, img_size, scale)
#
#         # Since batch size is one, convert variables to singular form
#         bbox = bboxes[0]
#         label = labels[0]
#         rpn_score = rpn_scores[0]
#         rpn_loc = rpn_locs[0]
#         roi = rois
#
#         # Sample RoIs and forward
#         sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
#             roi, bbox, label,
#             self.loc_normalize_mean, self.loc_normalize_std)
#         sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
#         roi_cls_loc, roi_score = self.faster_rcnn.head(
#             features, sample_roi, sample_roi_index)
#
#         # RPN losses
#         gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
#             bbox, anchor, img_size)
#         rpn_loc_loss = _fast_rcnn_loc_loss(
#             rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
#         rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)
#
#         # Losses for outputs of the head.
#         n_sample = roi_cls_loc.shape[0]
#         roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
#         roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]
#         roi_loc_loss = _fast_rcnn_loc_loss(
#             roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
#         roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)
#
#         loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
#         chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
#                                  'rpn_cls_loss': rpn_cls_loss,
#                                  'roi_loc_loss': roi_loc_loss,
#                                  'roi_cls_loss': roi_cls_loss,
#                                  'loss': loss},
#                                 self)
#         return loss
#
#
# def _smooth_l1_loss(x, t, in_weight, sigma):
#     sigma2 = sigma ** 2
#     diff = in_weight * (x - t)
#     abs_diff = F.absolute(diff)
#     flag = (abs_diff.array < (1. / sigma2)).astype(np.float32)
#
#     y = (flag * (sigma2 / 2.) * F.square(diff) +
#          (1 - flag) * (abs_diff - 0.5 / sigma2))
#
#     return F.sum(y)
#
#
# def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
#     xp = chainer.backends.cuda.get_array_module(pred_loc)
#
#     in_weight = xp.zeros_like(gt_loc)
#     # Localization loss is calculated only for positive rois.
#     in_weight[gt_label > 0] = 1
#     loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
#     # Normalize by total number of negtive and positive rois.
#     loc_loss /= xp.sum(gt_label >= 0)
#     return loc_loss
