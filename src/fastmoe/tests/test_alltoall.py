import os
import sys
import json
import math
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fmoe.functions import ensure_comm
from fmoe.layers import _fmoe_general_global_forward as naive_fwd
from fmoe.transformer import FMoETransformerMLP
from fmoe.gates import FileGate
import module


def _ensure_initialized():
    if 'RANK' not in os.environ:
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

def _test_exchange(tot_experts, d_model, new_mapping):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    if rank == 0:
        print(f"test exchange {world_size} {tot_experts} {d_model} {new_mapping}")

    num_expert = tot_experts // world_size

    params = []
    for _ in range(num_expert):
        m = torch.rand(d_model, d_model * 4, dtype=torch.float16).cuda()
        params.append(m)
    
    torch.cuda.synchronize()
    begin = time.time()

    my_send = []
    my_recv = []
    send_buffer = torch.zeros_like(params[0])
    recv_list = []
    for idx in range(tot_experts):
        if idx // num_expert == new_mapping[idx]:
            continue
        if idx // num_expert == rank:
            local_idx = idx % num_expert
            to_rank = new_mapping[idx] // num_expert
            to_local_idx = new_mapping[idx] % num_expert
            
            my_send.append((local_idx,to_rank,to_local_idx))
            send_buffer.copy_(params[local_idx])
            if to_rank != rank:
                torch.distributed.isend(send_buffer, to_rank).wait()

        if new_mapping[idx] // num_expert == rank:
            from_local_idx = idx % num_expert
            from_rank = idx // num_expert
            local_idx = new_mapping[idx] % num_expert
            my_recv.append((local_idx, from_rank, from_local_idx))
            recv_buffer = torch.zeros_like(send_buffer)
            if from_rank != rank:
                torch.distributed.irecv(recv_buffer, from_rank).wait()
            else:
                recv_buffer = send_buffer.data.clone()
                recv_list.append(recv_buffer)

    for recv_idx in range(len(recv_list)):
        params[my_recv[recv_idx][0]].copy_(recv_list[recv_idx])

    torch.cuda.synchronize() 
    end = time.time()

    if rank == 0:
        print(f"T={end - begin:.3f}")


if __name__ == '__main__':
    tot_experts = int(sys.argv[1])
    d_model = int(sys.argv[2])

    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert tot_experts % world_size == 0
    num_expert = tot_experts // world_size

    def gen_new_mapping(cnt):
        import random
        random.seed(42)
        new_mapping = [ i // num_expert for i in range(tot_experts)]

        for i in range(cnt):
            x = i % tot_experts
            y = (num_expert * i) % tot_experts
            a = new_mapping[x]
            new_mapping[x] = new_mapping[y]
            new_mapping[y] = a

        return new_mapping

    for _ in range(5):
        _test_exchange(tot_experts, d_model, gen_new_mapping(1))
        _test_exchange(tot_experts, d_model, gen_new_mapping(2))
        _test_exchange(tot_experts, d_model, gen_new_mapping(4))
        _test_exchange(tot_experts, d_model, gen_new_mapping(8))
        _test_exchange(tot_experts, d_model, gen_new_mapping(16))
        _test_exchange(tot_experts, d_model, gen_new_mapping(32))
