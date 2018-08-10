#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import OrderedDict

from dataset import SatelliteImages, SubSatelliteImages
from util import encode, decode, show, collate, metric_to_img

import numpy as np
import argparse
import sys


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
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.conv3_drop(x)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = F.relu(F.max_pool2d(self.conv6(x), 2))
        x = self.fc(x)
        return torch.sigmoid(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, (exist_ship, segment)) in enumerate(train_loader):
        data, exist_shipm, segment = data.to(device), exist_ship.to(device), segment.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum(- exist_ship.float() * torch.log(output) - (1 - exist_ship).float() * torch.log(1 - output)) / exist_ship.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='1バッチあたり何枚か')
    parser.add_argument('--epochs', '-e', type=int, default=10,
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
                        help='使うlossの種類'),
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
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, *kwargs)
    test_loader = DataLoader(val_dataset,
                             batch_size=args.val_batch_size if hasattr(args, "val_batch_size") else 1000, *kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == "__main__":
    # dataset = SatelliteImages(".")
    # img, target = dataset[1]
    # show(img)
    # show(metric_to_img(target)[1])

    main()
