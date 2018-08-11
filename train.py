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
from datetime import datetime



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3)
        self.conv7 = nn.Conv2d(16, 8, kernel_size=3)
        self.fc = nn.Linear(8*4*4, 1)

    def forward(self, x):
        try:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
            x = self.conv3_drop(x)
            x = F.relu(F.max_pool2d(self.conv4(x), 2))
            x = F.relu(F.max_pool2d(self.conv5(x), 2))
            x = F.relu(F.max_pool2d(self.conv6(x), 2))
            x = F.relu(F.max_pool2d(self.conv7(x), 2))
            x = self.fc(x.view(x.size(0), -1)).squeeze()
        except:
            pass
        return torch.sigmoid(x)


def train(model, device, train_loader, optimizer, reporter, step):
    data, (exist_ship, segment) = next(iter(train_loader))
    model.train()
    map(lambda x: x.to(device), [data, exist_ship, segment])
    optimizer.zero_grad()
    output = model(data)
    loss = torch.sum(- exist_ship.float() * torch.log(output) - (1 - exist_ship).float() * torch.log(1 - output)) / exist_ship.size(0)
    loss.backward()
    optimizer.step()

    pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    correct = (pred == exist_ship.view_as(pred).float()).sum().float() / output.size(0)
    reporter.report({"train/loss": loss.item(),
                     "train/accuracy": correct.item()})
    reporter.writer.add_scalar("train/loss", loss.item(), step)
    reporter.writer.add_pr_curve("train", exist_ship, pred, step)


def eval(model, device, eval_loader, reporter, step):
    model.eval()
    eval_loss = 0
    ground_true = []
    predict = []
    with torch.no_grad():
        for data, (exist_ship, segment) in eval_loader:
            map(lambda x: x.to(device), [data, exist_ship, segment])
            output = model(data)
            eval_loss += (torch.sum(- exist_ship.float() * torch.log(output) - (1 - exist_ship).float() * torch.log(1 - output)) / exist_ship.size(0)).item()  # sum up batch loss
            predict.append(torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)))
            ground_true.append(exist_ship)

    eval_loss /= len(eval_loader)
    ground_true = torch.cat(ground_true).view(-1)
    predict = torch.cat(predict).view(-1)

    reporter.report({"eval/loss": eval_loss.item(),
                     "eval/accuracy": (ground_true == predict).sum().float().mean().item()})
    reporter.writer.add_scalar("eval/loss", eval_loss.item(), step)
    reporter.writer.add_pr_curve("eval", ground_true, predict, step)



def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
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
    parser.add_argument('--eval_interval', '-ei', type=int, default=1,
                        help='検証をどの周期で行うか')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(0)

    dataset = SubSatelliteImages(".", train=True,
                                 transform=transforms.Compose(
                                     (transforms.ToTensor(),
                                      transforms.Normalize((0.2047899,  0.2887916, 0.3172972),
                                                           (0.03384494, 0.02707603, 0.01996508)))),
                                 target_transform=transforms.Compose(
                                     (metric_to_img, )
                                 ))
    train_dataset, eval_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, *kwargs)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size if hasattr(args, "val_batch_size") else 5,
                             collate_fn=collate, *kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter(os.path.join(args.out, datetime.now().strftime('%m-%d_%H:%M:%S')))
    reporter = Repoter(['epoch', 'iteration', 'train/loss', 'train/accuracy', 'eval/loss', 'eval/accuracy', 'elapsed_time'], writer,
                       all_epoch=args.epochs, iter_per_epoch=len(train_loader) // args.batch_size + 1)

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
        for iteration_idx in range(1, len(train_loader) // args.batch_size + 2):
            step = iteration_idx + (epoch_idx - 1) * (len(train_loader) // args.batch_size)

            with reporter.scope(epoch_idx, iteration_idx, step):
                train(model, device, train_loader, optimizer, reporter, step)
                if step % args.eval_interval == 0:
                    eval(model, device, eval_dataset, reporter, step)
                # trainer(epoch_idx, cum_iteration)


if __name__ == "__main__":
    main()
