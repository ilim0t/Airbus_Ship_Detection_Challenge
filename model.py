#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return x0


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

        self.upsamp0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=3)
        self.outconv = nn.Conv2d(64+32, 1, kernel_size=1)

    def forward(self, x):  # => 3, 256
        x0 = self.down0(x)  # => 32, 256
        x1 = self.down1(F.max_pool2d(x0, 2))  # => 64, 128
        x2 = self.down2(F.max_pool2d(x1, 2))  # => 128, 64

        x3 = self.conv(F.max_pool2d(x2, 2))  # => 256, 32

        x2 = self.up1(torch.cat((x2, self.upsamp2(x3)), dim=1))  # => 128, 128
        x1 = self.up1(torch.cat((x1, self.upsamp1(x2)), dim=1))  # => 64, 256

        x0 = self.outconv(torch.cat((x0, self.upsamp0(x1)), dim=1))  # => 1, 768
        x0 = torch.sigmoid(x0)
        return x0
