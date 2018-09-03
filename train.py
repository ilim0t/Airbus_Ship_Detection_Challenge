#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from collections import OrderedDict

from dataset import SatelliteImages
from util import encode, decode, show, collate
from repoter import Repoter
from tensorboardX import SummaryWriter
from model import *

import numpy as np
import argparse
import sys
import os
import time
import itertools
import functools
import json
from datetime import datetime


def train(model, device, train_loader, optimizer, reporter, step):
    data, (bboxs, masks) = next(train_loader)
    model.train()
    data = data.to(device)
    for masks in masks:
        for mask in masks:
            mask = mask.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.binary_cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    predict = output.data > 0.5
    # target = target.data > 0.5

    IoU = torch.sum(predict == target, dim=(1, 2)).float() / torch.sum((predict + target) > 1, dim=(1, 2)).float()
    threshold = torch.arange(0.5, 1, 0.05)

    accuracy = (predict == target).float().mean()
    reporter.report({"train/loss": loss.item(),
                     "train/accuracy": accuracy.item()})
    reporter.writer.add_scalar("train/loss", loss.item(), step)
    reporter.writer.add_scalar("train/accuracy", accuracy.item(), step)
    # reporter.writer.add_pr_curve("train", exist_ship, output.squeeze(), step)
    reporter.writer.add_image('Image', utils.make_grid(data[:2], normalize=True, scale_each=True), step)


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

    def convert(bbox, mask):
        if bbox is None:
            return bbox, mask
        else:
            return [torch.tensor(i, dtype=torch.float) for i in bbox],\
                   [torch.tensor(i, dtype=torch.float) for i in mask]

    # dataset = SatelliteImages(".", train=True,
    dataset = SatelliteImages(
        ".", train=True, transform=transforms.Compose((
            # transforms.Resize(384),
            transforms.Resize(256),
            #  lambda img: img.reshape((*img.shape, 1)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.2047899,  0.2887916, 0.3172972),
            #     (0.03384494, 0.02707603, 0.01996508)))
        )), mask_transform=transforms.Compose((
            decode,
            # lambda x: ([torch.tensor(i, dtype=torch.float) for i in x[0]], [torch.tensor(j, dtype=torch.float) for j in x[1]])
            # transforms.ToPILImage(),
            transforms.Resize(256),
            # transforms.ToTensor()
            )
        ), on_server=(sys.platform == "linux"))

    n = min(64*4, int(len(dataset) * 0.01))
    train_dataset, eval_dataset = random_split(dataset, (len(dataset) - n, n))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, **kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size if hasattr(args, "val_batch_size") else 5,
                             collate_fn=collate, **kwargs)
    # model = Net().to(device)
    model = MaskRCNN().to(device)
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
                if step % args.eval_interval == 0:
                    # return
                    eval(model, device, eval_loader, reporter, step)
        torch.save(model.state_dict(), "snapshot")


if __name__ == "__main__":
    main()
