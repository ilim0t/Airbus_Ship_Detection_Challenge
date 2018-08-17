#!/usr/bin/env python

import chainer
import chainer.links as L
import chainer.functions as F

from collections import OrderedDict

from dataset import SatelliteImages, SubSatelliteImages
from tensorboardX import SummaryWriter
from model import Net, Unet, Unet2
from util import decode, Normalize

import numpy as np
import argparse
import sys
import os
import time
import itertools
import functools
import json
from datetime import datetime


class Net(chainer.Chain):
    def __init__(self, model):
        super(Net, self).__init__()
        with self.init_scope():
            self.model = model

    def forward(self, x):
        x = self.model(x)
        x = F.resize_images(x, (768, 768))
        return x

    def __call__(self, x, t):
        output = self.forward(x)
        predict = output.data > 0.5

        accuracy = F.average((predict == t).astype(np.float32))
        loss = F.sigmoid_cross_entropy(output, t.astype(np.int32))

        chainer.reporter.report({'loss': loss}, self)
        chainer.reporter.report({'accuracy': accuracy}, self)
        return loss


class Writer(chainer.training.Extension):
    def __init__(self, writer=SummaryWriter(), log_report="LogReport"):
        self._writer = writer
        self._log_report = log_report

    def __call__(self, trainer):
        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, chainer.training.extensions.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        # for key, value in log[-1].items():
        for key, value in trainer.observation.items():
                self._writer.add_scalar(key, value, trainer.updater.iteration)


def main():
    parser = argparse.ArgumentParser(description='Airbus Ship Detection Challenge')
    parser.add_argument('--batch_size', '-b', type=int, default=2,
                        help='1バッチあたり何枚か')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='何epochやるか')
    parser.add_argument('--out', '-o', default='../result',
                        help='結果を出力するディレクトリ')
    parser.add_argument('--resume', '-r', default='',
                        help='指定したsnapshotから継続して学習します')
    parser.add_argument('--frequency', '-f', type=int, default=1,
                        help='指定したepochごとに重みを保存します')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID')
    parser.add_argument('--total_photo_num', '-n', type=int, default=-1,
                        help='使用する写真データの数'),  # (9815, 39269)
    parser.add_argument('--log_interval', '-i', type=int, default=1,
                        help='何iteraionごとに画面に出力するか')
    parser.add_argument('--model', '-m', type=int, default=0,
                        help='使うモデルの種類')
    parser.add_argument('--lossfunc', '-l', type=int, default=0,
                        help='使うlossの種類')
    parser.add_argument('--eval_interval', '-ei', type=int, default=200,
                        help='検証をどの周期で行うか')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    model = Net(Unet2())
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    dataset = SubSatelliteImages("..",
                                 transform=chainer.Sequential(
                                     lambda img: img.resize((386, 386)),
                                     lambda img: np.asarray(img, dtype=np.float32).transpose((2, 0, 1)),
                                     Normalize(
                                         (0.2047899, 0.2887916, 0.3172972),
                                         (0.03384494, 0.02707603, 0.01996508))
                                 ),
                                 target_transform=chainer.Sequential(
                                     decode,
                                     lambda x: x.reshape((1, *x.shape))
                                 ))
    train, test = chainer.datasets.split_dataset_random(dataset, len(dataset) - min(64*4, int(len(dataset) * 0.01)), seed=0)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.eval_batch_size if hasattr(args, "val_batch_size") else 5,
                                                 repeat=False, shuffle=False)

    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(chainer.training.extensions.snapshot(), trigger=(frequency, 'epoch'))

    trainer.extend(chainer.training.extensions.LogReport(trigger=(1, "iteration")))

    if chainer.training.extensions.PlotReport.available():
        trainer.extend(
            chainer.training.extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(
            chainer.training.extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))

    writer = SummaryWriter(os.path.join(args.out, datetime.now().strftime('%m-%d_%H:%M:%S_bs-{}'.format(args.batch_size))))
    trainer.extend(Writer(writer))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == "__main__":
    main()
