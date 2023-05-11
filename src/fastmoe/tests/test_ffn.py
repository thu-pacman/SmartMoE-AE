import pytest

import os
import sys
import json
import math
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from test_ddp import _ensure_initialized

class Linear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True,
                 init_method=init.xavier_normal_, stride=1,
                 skip_bias_add=False):
        super().__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        self.weight = Parameter(torch.empty(
            self.output_size, self.input_size,
            device=torch.cuda.current_device(), dtype=torch.float16))
        init_method(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(
                self.output_size, device=torch.cuda.current_device(),
                dtype=torch.float16))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # Matrix multiply.
        output = F.linear(input, self.weight)
        # All-reduce across all the partitions.
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output = output
            output_bias = self.bias
        return output

class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, d_model):
        super().__init__()

        # Project to 4h.
        self.dense_h_to_4h = Linear(
            d_model,
            d_model * 4)

        self.activation_func = F.gelu

        # Project back to h.
        # self.dense_4h_to_h = mpu.CCRowParallelLinear(
        self.dense_4h_to_h = Linear(
            d_model * 4,
            d_model)


    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

def _test_ffn(num_layers, d_model, seq_len, global_batch_size, micro_batch_size):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    num_micro_batch = global_batch_size // micro_batch_size // world_size

    m = []
    for _ in range(num_layers):
        m1 = MLP(d_model) 
        m.append(m1)

    n_run = 10
    for _ in range(n_run):
        x1 = torch.rand(seq_len * micro_batch_size, d_model, dtype=torch.float16).cuda()
        x1.requires_grad = True
        y1 = m[0](x1)
        y1.sum().backward()
        #print(x1.shape, y1.shape)

    for round in range(n_run):
        loss = torch.zeros(1,dtype=torch.float16,device='cuda')
        t_fwd = 0
        t_bwd = 0
        for _ in range(num_micro_batch):
            x1 = torch.rand(seq_len * micro_batch_size, d_model, dtype=torch.float16).cuda()
            x1.requires_grad = True
            torch.cuda.synchronize()
            t1_begin = time.time()
            for i in range(num_layers):
                x1 = m[i](x1)
            torch.cuda.synchronize()
            t1_end = time.time()
            loss = x1.sum()
            t_fwd += t1_end - t1_begin
            torch.cuda.synchronize()
            t2_begin = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t2_end = time.time()
            t_bwd += t2_end - t2_begin

        if rank == 0:
            print(f"round {round} t_fwd={t_fwd:.3f} t_bwd={t_bwd:.3f}", flush=True)

if __name__ == '__main__':
    _test_ffn(24, 1536, 1024, 256, 2)
