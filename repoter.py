#!/usr/bin/env python
from tensorboardX import SummaryWriter
from chainer.training.extensions import LogReport

import os
import sys
import datetime
from typing import Dict
import time
from tensorboardX import SummaryWriter



# class Repoter(SummaryWriter, LogReport):
class Repoter(object):
    _out = sys.stdout

    def __init__(self, entries, writer: SummaryWriter, trigger=(1, 'epoch'), all_epoch=None, iter_per_epoch=None, out="result"):
        self.out = out
        self.log = [{}]
        self._log_len = 0
        self.writer = writer
        self._epoch, self._iteration, self._cum_iteration = 0, 0, 0
        self._start = time.time()

        entry_widths = [max(10, len(s)) for s in entries]
        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._out.write(header)

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

        self._status_template = None
        self._bar_length = 50
        self.training_length = (all_epoch, "epoch")
        self.iter_per_epoch = iter_per_epoch
        self._recent_timing = []

    def report(self, values: Dict[str, float]):
        self.log[self._log_len].update(values)
        # for key, value in values.items():
        #     self.writer.add_scalar(key, value, self._cum_iteration)

    def scope(self, epoch, iteration, cum_iteration):
        self._epoch = epoch
        self._iteration = iteration
        self._cum_iteration = cum_iteration
        return self

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log[-1].update({
            "epoch": self._epoch,
            "iteration": self._iteration,
            "elapsed_time": time.time() - self._start
        })
        self.print()
        self.progressbar()
        self.log.append({})
        return False

    def print(self):
        out = self._out
        log = self.log
        log_len = self._log_len
        while len(log) > log_len:
            out.write('\033[J')
            observation = log[log_len]
            for entry, template, empty in self._templates:
                if entry in observation:
                    out.write(template.format(observation[entry]))
                else:
                    out.write(empty)
            out.write('\n')
            log_len += 1
        self._log_len = log_len

    def progressbar(self):
        # initialize some attributes at the first call
        training_length = self.training_length

        stat_template = self._status_template
        if stat_template is None:
            stat_template = self._status_template = (
                '{iteration:10} iter, {epoch} epoch / %s %ss\n' %
                training_length)

        length, unit = training_length
        out = self._out
        epoch = self._epoch
        iteration = self._iteration
        percent_epoch = epoch - 1 + iteration / self.iter_per_epoch
        recent_timing = self._recent_timing
        now = time.time()

        recent_timing.append((iteration, percent_epoch, now))

        out.write('\033[J')

        if unit == 'iteration':
            rate = iteration / length
        else:
            rate = percent_epoch / length
        rate = min(rate, 1.0)

        bar_length = self._bar_length
        marks = '#' * int(rate * bar_length)
        out.write('     total [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), rate))

        epoch_rate = percent_epoch - int(percent_epoch)
        marks = '#' * int(epoch_rate * bar_length)
        out.write('this epoch [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), epoch_rate))

        status = stat_template.format(iteration=iteration, epoch=epoch)
        out.write(status)

        old_t, old_e, old_sec = recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (percent_epoch - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')

        if unit == 'iteration':
            estimated_time = (length - iteration) / speed_t
        else:
            estimated_time = (length - percent_epoch) / speed_e
        estimated_time = max(estimated_time, 0.0)
        out.write('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                  .format(speed_t,
                          datetime.timedelta(seconds=estimated_time)))

        # move the cursor to the head of the progress bar
        out.write('\033[4A')
        out.flush()

        if len(recent_timing) > 100:
            del recent_timing[0]
