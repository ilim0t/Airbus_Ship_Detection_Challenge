#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from collections import OrderedDict

from dataset import SatelliteImages, SubSatelliteImages
from util import encode, decode, show, collate, metric_to_img
from trainer import Trainer, MyTrainer
import extensions
from repoter import Repoter
from tensorboardX import SummaryWriter


import numpy as np
import argparse
import sys
import os
import time
import itertools
import functools
import json
from datetime import datetime



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


def train(model, device, train_loader, optimizer, reporter, step):
    data, (exist_ship, segment) = next(train_loader)
    model.train()
    map(lambda x: x.to(device), [data, exist_ship, segment])
    optimizer.zero_grad()
    output = model(data)
    # loss = torch.sum(- exist_ship.float() * torch.log(output) - (1 - exist_ship).float() * torch.log(1 - output)) / exist_ship.size(0)
    loss = F.cross_entropy(torch.cat((1-output, output), 1), exist_ship)
    loss.backward()
    optimizer.step()

    pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).squeeze()
    accuracy = (pred == exist_ship.view_as(pred).float()).sum().float() / output.size(0)
    reporter.report({"train/loss": loss.item(),
                     "train/accuracy": accuracy.item()})
    reporter.writer.add_scalar("train/loss", loss.item(), step)
    reporter.writer.add_scalar("train/accuracy", accuracy.item(), step)
    reporter.writer.add_pr_curve("train", exist_ship, output.squeeze(), step)
    reporter.writer.add_image('Image', utils.make_grid(data[:2], normalize=True, scale_each=True), step)


def eval(model, device, eval_loader, reporter, step):
    model.eval()
    eval_loss = 0
    ground_true = []
    predict = []
    cum_output = []
    with torch.no_grad():
        for data, (exist_ship, segment) in eval_loader:
            map(lambda x: x.to(device), [data, exist_ship, segment])
            output = model(data)
            # eval_loss += (torch.sum(- exist_ship.float() * torch.log(output) - (1 - exist_ship).float() * torch.log(1 - output)) / exist_ship.size(0)).item()  # sum up batch loss
            eval_loss += F.cross_entropy(torch.cat((1 - output, output), 1), exist_ship)
            predict.append(torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).squeeze())
            ground_true.append(exist_ship)
            cum_output.append(output)

    eval_loss /= len(eval_loader)
    ground_true = torch.cat(ground_true).view(-1)
    predict = torch.cat(predict).view(-1)
    cum_output = torch.cat(cum_output).view(-1)
    accuracy = (ground_true.float() == predict).float().mean()

    reporter.report({"eval/loss": eval_loss.item(),
                     "eval/accuracy": accuracy.item()})
    reporter.writer.add_scalar("eval/loss", eval_loss.item(), step)
    reporter.writer.add_scalar("eval/accuracy", accuracy.item(), step)
    reporter.writer.add_pr_curve("eval", ground_true, cum_output, step)


def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='1バッチあたり何枚か')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='何epochやるか')
    parser.add_argument('--out', '-o', default='result',
                        help='結果を出力するディレクトリ')
    parser.add_argument('--resume', '-r', default='',
                        help='指定したsnapshotから継続して学習します')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='指定したepochごとに重みを保存します')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='GPUを使用するか')
    parser.add_argument('--total_photo_num', '-n', type=int, default=-1,
                        help='使用する写真データの数'),  # (9815, 39269)
    parser.add_argument('--log_interval', '-i', type=int, default=1,
                        help='何iteraionごとに画面に出力するか')
    parser.add_argument('--model', '-m', type=int, default=0,
                        help='使うモデルの種類')
    parser.add_argument('--lossfunc', '-l', type=int, default=0,
                        help='使うlossの種類')
    parser.add_argument('--eval_interval', '-ei', type=int, default=20,
                        help='検証をどの周期で行うか')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)

    # dataset = SatelliteImages(".", train=True,
    dataset = SubSatelliteImages(".", train=True,
                              transform=transforms.Compose((
                                  transforms.Resize(384),
                                  # transforms.Resize(192),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.2047899,  0.2887916, 0.3172972),
                                      (0.03384494, 0.02707603, 0.01996508))
                              )),
                              target_transform=transforms.Compose((
                                  metric_to_img,
                              )))

    n = min(64*4, int(len(dataset) * 0.01))
    train_dataset, eval_dataset = random_split(dataset, (len(dataset) - n, n))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, *kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size if hasattr(args, "val_batch_size") else 5,
                             collate_fn=collate, *kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(os.path.join(args.out, datetime.now().strftime('%m-%d_%H:%M:%S')))
    reporter = Repoter(['epoch', 'iteration', 'train/loss', 'train/accuracy', 'eval/loss', 'eval/accuracy', 'elapsed_time'], writer,
                       all_epoch=args.epochs, iter_per_epoch=len(train_loader))

    # trainer = Trainer(model=model, optimizer=optimizer, device=device, writer=writer,
    #                   train_loader=train_loader, eval_loader=eval_loader)
    # trainer.extend([
    #     extensions.Extension(trigger=(100, "iteration"), func=eval),
    #     extensions.PrintReport(
    #         ['epoch', 'iter', 'main/loss', 'eval/loss', 'eval/correct', 'elapsed_time']),
    #     extensions.LogReport()
    #     extensions.Extension(trigger=(3, "iteration"), name="writter",
    #                          func=functools.partial(writer.export_scalars_to_json, "./all_scalars.json"),
    #                          finalize=writer.close)
    # ])

    for epoch_idx in range(1, args.epochs + 1) if args.epochs > 0 else itertools.count(1):
        train_iter = iter(train_loader)
        for iteration_idx in range(1, len(train_loader) + 1):
            step = iteration_idx + (epoch_idx - 1) * len(train_loader)

            with reporter.scope(epoch_idx, step, step):
                train(model, device, train_iter, optimizer, reporter, step)
                if step % args.eval_interval == 0:
                    # return
                    eval(model, device, eval_loader, reporter, step)
                # trainer(epoch_idx, cum_iteration)


if __name__ == "__main__":
    main()
