# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron global variables."""

import os
import sys
import time

import torch

from megatron.tokenizer import build_tokenizer
from .arguments import parse_args
from .microbatches import build_num_microbatches_calculator

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def set_global_variables(extra_args_provider=None, args_defaults={},
                         ignore_unknown_args=False):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""
    args = _parse_args(extra_args_provider=extra_args_provider,
                       defaults=args_defaults,
                       ignore_unknown_args=ignore_unknown_args)
    _build_num_microbatches_calculator(args)
    if args.vocab_file:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_adlr_autoresume(args)
    _set_timers()


def _parse_args(extra_args_provider=None, defaults={},
                ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = parse_args(extra_args_provider=extra_args_provider,
                              defaults=defaults,
                              ignore_unknown_args=ignore_unknown_args)
    return _GLOBAL_ARGS


def _build_num_microbatches_calculator(args):

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.times = []

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        duration = (time.time() - self.start_time)
        self.elapsed_ += duration
        self.times.append(duration)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.times = []
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        total_elapsed_ = self.elapsed_
        record_cnt = len(self.times)
        if record_cnt > 0 :
            mean_elapsed = total_elapsed_ / record_cnt
            min_elapsed = min(self.times)
            max_elapsed = max(self.times)
        else :
            mean_elapsed = total_elapsed_
            min_elapsed = total_elapsed_
            max_elapsed = total_elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return total_elapsed_, mean_elapsed, min_elapsed, max_elapsed, record_cnt


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            total_elapsed, _, _, _, _ = self.timers[name].elapsed(reset=reset)
            value = total_elapsed / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names, normalizer=1.0, iteration=-1, reset=True):
        """Log a group of timers."""
        from megatron.mpu import get_data_parallel_rank
        from megatron.mpu import get_tensor_model_parallel_rank
        from megatron.mpu import get_pipeline_model_parallel_rank        
        assert normalizer > 0.0
        assert torch.distributed.is_initialized(), "torch distributed not initialized."
        
        rank = torch.distributed.get_rank()
        dp_rank = get_data_parallel_rank()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()
        head_string = '"type":"time", "iteration": {}, "rank": {}, "tp_rank": {}, "pp_rank": {}, "dp_rank": {}'.format(
            iteration, rank, tp_rank, pp_rank, dp_rank)
        metrics = ""
        has_metrics = False
        
        for name in names:
            total_elapsed, mean_elapsed, min_elapsed, max_elapsed, record_cnt = self.timers[name].elapsed(reset=reset)
            elapsed_time = total_elapsed * 1000.0 / normalizer
            mean_elapsed = mean_elapsed * 1000.0 / normalizer
            min_elapsed = min_elapsed * 1000.0 / normalizer
            max_elapsed = max_elapsed * 1000.0 / normalizer
            cur_string = '"total": {:.2f}, "avg": {:.2f}, "max": {:.2f}, "min": {:.2f}, "cnt": {}'.format(
                elapsed_time, mean_elapsed, min_elapsed, max_elapsed, record_cnt)
            cur_string = '"{}":'.format(name) + "{" + cur_string + "}"
            if has_metrics :
                metrics += ", "
            has_metrics = True
            metrics += cur_string
        
        head_string += ', "perf": ' + '{' + metrics + '}'
        string = "{" + head_string + "}"
        
        import os
        log_prefix = os.environ['PROFILER_LOG_PATH']
        file_name = log_prefix + '/{}.profiler'.format(torch.distributed.get_rank())
        
        with open(file_name, "a") as f :
            f.write(string + '\n')            

class TimerOP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timer_name, is_front):
        # if dist.get_rank() == 0:
        #     print("call farward timer" + timer_name + str(is_front))
        timers = get_timers()
        if is_front:
            timers(timer_name + '_fwd').start()
        else:
            timers(timer_name + '_fwd').stop()
        
        ctx.timer_name = timer_name
        ctx.is_front = is_front

        return input

    @staticmethod
    def backward(ctx, grad_in):
        # if dist.get_rank() == 0:
        #     print("call backward timer" + ctx.timer_name + str(ctx.is_front))
        timers = get_timers()
        timer_name = ctx.timer_name + '_bwd'
        if ctx.is_front:
            timers(timer_name).stop()
        else:
            timers(timer_name).start()
        
        return grad_in, None, None