#!/usr/bin/env python

from typing import Callable, Union, Tuple
import sys
import os


def get_trigger(period: int, unit: str):
    assert unit in ["epoch", "iteration", "iter"], "Unkwown unit: " + str(unit)
    if unit == "epoch":
        def trigger(epoch: int):
            if epoch % period == 0:
                return True
            else:
                return False
    else:
        def trigger(iteration: int):
            if iteration % period == 0:
                return True
            else:
                return False
    return trigger


def never_fire_trigger():
    return False


class Extension(object):
    def __init__(self, trigger: Union[Callable[..., bool], Tuple]= None, name: str=None, func: Callable=None, initializer: Callable=None, finalize: Callable=None):
        assert func is not None or trigger is None, "func, trigger のどちらかを設定してください"

        if callable(trigger):
            self.trigger = trigger
        elif isinstance(trigger, tuple):
            self.trigger = get_trigger(*trigger)
        else:
            self.trigger = never_fire_trigger

        if func:
            self.main = func
        if initializer:
            self.initializer = initializer
        if finalize:
            self.finalize = finalize

        self.name = name or (func or initializer or finalize).__name__

from chainer.training.extensions import PrintReport


class PrintReport(PrintReport):
    def main(self, writer):
        out = sys.stdout

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = writer.scalar_dict
        return

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            # delete the printed contents from the current cursor
            if os.name == 'nt':
                util.erase_console(0, 0)
            else:
                out.write('\033[J')
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

    def trigger(self, iteration):
        return iteration % 5 == 0


from chainer.training.extensions import LogReport


class LogReport(LogReport):
    def main(self):
        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer):
            # output the result
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater
            stats_cpu['epoch'] = updater.epoch
            stats_cpu['iteration'] = updater.iteration
            stats_cpu['elapsed_time'] = trainer.elapsed_time

            if self._postprocess is not None:
                self._postprocess(stats_cpu)

            self._log.append(stats_cpu)

            # write to the log file
            if self._log_name is not None:
                log_name = self._log_name.format(**stats_cpu)
                with utils.tempdir(prefix=log_name, dir=trainer.out) as tempd:
                    path = os.path.join(tempd, 'log.json')
                    with open(path, 'w') as f:
                        json.dump(self._log, f, indent=4)

                    new_path = os.path.join(trainer.out, log_name)
                    shutil.move(path, new_path)

            # reset the summary for the next output
            self._init_summary()

    def trigger(self):
        return True
