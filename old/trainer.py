#!/usr/bin/env python

from chainer.training import Trainer

import collections
import os
import sys
import time
import traceback

import six
from inspect import signature

from chainer import reporter as reporter_module
from chainer import serializer as serializer_module
from chainer.training import extension as extension_module
from chainer.training import trigger as trigger_module
from chainer.utils import argument


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class _ExtensionEntry(object):

    def __init__(self, extension, priority, trigger):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority


class MyTrainer(Trainer):
    def __init__(self, updater, stop_trigger=None, out='result',
                 extensions=None):
        self.updater = updater  # contains train_iter, optimizer, device
        self.stop_trigger = trigger_module.get_trigger(stop_trigger)
        self.observation = {}
        self.out = out
        if extensions is None:
            extensions = []

        reporter = reporter_module.Reporter()
        for name, optimizer in six.iteritems(updater.get_all_optimizers()):
            reporter.add_observer(name, optimizer.target)
            reporter.add_observers(
                name, optimizer.target.namedlinks(skipself=True))
        self.reporter = reporter

        self._done = False
        self._extensions = collections.OrderedDict()

        self._start_at = None
        self._snapshot_elapsed_time = 0.0
        self._final_elapsed_time = None

        updater.connect_trainer(self)
        for ext in extensions:
            self.extend(ext)

    @property
    def elapsed_time(self):
        if self._done:
            return self._final_elapsed_time
        if self._start_at is None:
            raise RuntimeError('training has not been started yet')
        return _get_time() - self._start_at + self._snapshot_elapsed_time

    def extend(self, extension, name=None, trigger=None, priority=None,
               **kwargs):
        argument.check_unexpected_kwargs(
            kwargs,
            invoke_before_training='invoke_before_training has been removed '
            'since Chainer v2.0.0. Use initializer= instead.')
        argument.assert_kwargs_empty(kwargs)

        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = getattr(
                extension, 'priority', extension_module.PRIORITY_READER)

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            extension, priority, trigger)

    def get_extension(self, name):
        extensions = self._extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError('extension %s not found' % name)

    def run(self, show_loop_exception_msg=True):
        if self._done:
            raise RuntimeError('cannot run training loop multiple times')

        try:
            os.makedirs(self.out)
        except OSError:
            pass

        # sort extensions by priorities
        extension_order = sorted(
            self._extensions.keys(),
            key=lambda name: self._extensions[name].priority, reverse=True)  # List[extension_name: str]
        extensions = [(name, self._extensions[name])
                      for name in extension_order]  # = List[Tuple[extension_name: str, trainer._ExtensionEntry]]

        self._start_at = _get_time()

        # invoke initializer of each extension
        for _, entry in extensions:
            initializer = getattr(entry.extension, 'initialize', None)
            if initializer:
                initializer(self)

        update = self.updater.update
        reporter = self.reporter
        stop_trigger = self.stop_trigger

        # main training loop
        try:
            while not stop_trigger(self):
                self.observation = {}
                with reporter.scope(self.observation):
                    update()
                    for name, entry in extensions:
                        if entry.trigger(self):
                            entry.extension(self)
        except Exception as e:
            if show_loop_exception_msg:
                # Show the exception here, as it will appear as if chainer
                # hanged in case any finalize method below deadlocks.
                print('Exception in main training loop: {}'.format(e),
                      file=sys.stderr)
                print('Traceback (most recent call last):', file=sys.stderr)
                traceback.print_tb(sys.exc_info()[2])
                print('Will finalize trainer extensions and updater before '
                      'reraising the exception.', file=sys.stderr)
            six.reraise(*sys.exc_info())
        finally:
            for _, entry in extensions:
                finalize = getattr(entry.extension, 'finalize', None)
                if finalize:
                    finalize()
            self.updater.finalize()

        self._final_elapsed_time = self.elapsed_time
        self._done = True


class Trainer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs.update(trainer=self)
        self.extensions = []
        self.done = False

    def extend(self, extensions):
        for extension in extensions:
            self.extensions.append(extension)

    def initializer(self):
        for extension in self.extensions:
            initializer = getattr(extension, "initializer", None)
            if initializer:
                initializer(**{i: self.kwargs.get(i) for i in signature(initializer).parameters if hasattr(self.kwargs, i)})

    def finalize(self):
        for extension in self.extensions:
            finalize = getattr(extension, "finalize", None)
            if finalize:
                finalize(**{i: self.kwargs.get(i) for i in signature(finalize).parameters if hasattr(self.kwargs, i)})

    def __call__(self, epoch, iteration):
        idx = {"epoch": epoch, "iteration": iteration}
        for extension in self.extensions:
            if hasattr(extension, "trigger"):
                if extension.trigger(**{i: {**self.kwargs, **idx}[i] for i in signature(extension.trigger).parameters if
                                        i in self.kwargs or i in idx}):
                    extension.main(**{i: {**self.kwargs, **idx}[i] for i in signature(extension.main).parameters if i in self.kwargs or i in idx})

    def __enter__(self):
        self.initializer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False
