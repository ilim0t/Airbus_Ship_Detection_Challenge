#!/usr/bin/env python
import chainer
import chainer.functions as F
import chainer.links as L


class Net(chainer.Chain):
    def __init__(self):
        super(Net, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, ksize=5, stride=2)
            self.conv2 = L.Convolution2D(32, 64, ksize=3)
            self.bn2 = L.BatchNormalization(64)
            self.conv3 = L.Convolution2D(64, 128, ksize=3)
            self.conv4 = L.Convolution2D(128, 64, ksize=3)
            self.bn4 = L.BatchNormalization(64)
            self.conv5 = L.Convolution2D(64, 32, ksize=3)
            self.conv6 = L.Convolution2D(32, 16, ksize=3)

            self.fc = L.Linear(16, 1)

    def forward(self, x):  # => 3, 384
        x = F.max_pooling_2d(self.conv1(x), 2)  # => 32, 95
        x = F.max_pooling_2d(self.conv2(F.relu(x)), 2)  # => 32, 46
        x = F.dropout(x, 0.3)
        x = self.bn2(x)
        x = F.max_pooling_2d(self.conv3(F.relu(x)), 2)  # => 32, 22
        x = F.max_pooling_2d(self.conv4(F.relu(x)), 2)  # => 32, 10
        x = F.dropout(x, 0.3)
        x = self.bn4(x)
        x = F.max_pooling_2d(self.conv5(F.relu(x)), 2)  # => 32, 4
        x = F.max_pooling_2d(self.conv6(F.relu(x)), 2)  # => 32, 1

        x = self.fc(x.reshape(x.shape[0], -1))
        return F.sigmoid(x)


class Unet(chainer.Chain):
    def __init__(self):
        super(Unet, self).__init__()
        with self.init_scope():
            self.down0 = UnetBlock(3, 32)
            self.down1 = UnetBlock(32, 64)
            self.down2 = UnetBlock(64, 128)
            self.down3 = UnetBlock(128, 256)

            self.conv = chainer.Sequential(
                L.Convolution2D(256, 512, ksize=3, pad=1),
                F.relu,
                L.Convolution2D(512, 512, ksize=3, pad=1)
            )

            self.upsamp3 = L.Deconvolution2D(512, 256, ksize=2, stride=2)
            self.up3 = UnetBlock(512, 256)

            self.upsamp2 = L.Deconvolution2D(256, 128, ksize=2, stride=2)
            self.up2 = UnetBlock(256, 128)

            self.upsamp1 = L.Deconvolution2D(128, 64, ksize=2, stride=2)
            self.up1 = UnetBlock(128, 64)

            self.upsamp0 = L.Deconvolution2D(64, 64, ksize=3, stride=3)
            self.outconv = L.Convolution2D(64+32, 1, ksize=1)

    def forward(self, x):  # => 3, 768 (=2^8*3)
        x0 = self.down0(x)  # => 32, 512
        x1 = self.down1(F.max_pooling_2d(x0, 3))  # => 64, 256
        x2 = self.down2(F.max_pooling_2d(x1, 2))  # => 128, 128
        x3 = self.down3(F.max_pooling_2d(x2, 2))  # => 256, 64

        x4 = self.conv(F.max_pooling_2d(x3, 2))  # => 512, 32

        x3 = self.up3(F.concat((x3, self.upsamp3(x4)), axis=1))  # => 256, 64
        x2 = self.up2(F.concat((x2, self.upsamp2(x3)), axis=1))  # => 128, 128
        x1 = self.up1(F.concat((x1, self.upsamp1(x2)), axis=1))  # => 64, 256

        x0 = self.outconv(F.concat((x0, self.upsamp0(x1)), axis=1))  # => 1, 768
        x0 = F.sigmoid(x0)
        return x0


class UnetBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1):
        super(UnetBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad)
            self.conv3 = L.Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        return x


class Unet2(Unet):
    def __init__(self):
        super(Unet, self).__init__()
        with self.init_scope():
            self.down0 = UnetBlock(3, 32)
            self.down1 = UnetBlock(32, 64)
            self.down2 = UnetBlock(64, 128)

            self.conv = chainer.Sequential(
                L.Convolution2D(128, 256, ksize=3, pad=1),
                F.relu,
                L.Convolution2D(256, 256, ksize=3, pad=1)
            )

            self.upsamp2 = L.Deconvolution2D(256, 128, ksize=2, stride=2)
            self.up2 = UnetBlock(256, 128)

            self.upsamp1 = L.Deconvolution2D(128, 64, ksize=2, stride=2)
            self.up1 = UnetBlock(128, 64)

            self.upsamp0 = L.Deconvolution2D(64, 64, ksize=2, stride=2)
            self.outconv = L.Convolution2D(64+32, 1, ksize=1)

    def forward(self, x):  # => 3, 256
        x0 = self.down0(x)  # => 32, 256
        x1 = self.down1(F.max_pooling_2d(x0, 2))  # => 64, 128
        x2 = self.down2(F.max_pooling_2d(x1, 2))  # => 128, 64

        x3 = self.conv(F.max_pooling_2d(x2, 2))  # => 256, 32

        x2 = self.up2(F.concat((x2, self.upsamp2(x3)), axis=1))  # => 128, 128
        x1 = self.up1(F.concat((x1, self.upsamp1(x2)), axis=1))  # => 64, 256

        x0 = self.outconv(F.concat((x0, self.upsamp0(x1)), axis=1))  # => 1, 768
        x0 = F.sigmoid(x0)
        return x0
