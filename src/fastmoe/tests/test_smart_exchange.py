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

smartmoe_last_mapping = {}

def _ensure_initialized():
    if 'RANK' not in os.environ:
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def load_gate_history(filename, layer_idx):
    rank = 0
    filename += "_rank{}.log".format(rank)

    with open(filename, "r") as f:
        lines = f.readlines()
        line = lines[layer_idx]
        _, table = line.split(':')
        microbatch_tables = json.loads(table)
        return microbatch_tables[0]


def _test_smart_exchange(spec, table_prefix, iter, enable_update, filename, layer_idx, history_lat, d_model, seq_len, batch_size, n_expert):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    if rank == 0:
        print(f"test smart exchange {filename} {layer_idx}")

    def file_gate(filename, layer_idx):
        def wrapper_gate(d_model, num_expert, world_size, top_k):
            return FileGate(d_model, num_expert, world_size, top_k, rank=rank, tensor_type=torch.float16, inp_shape=seq_len*batch_size, filename=filename, layer_idx=layer_idx)
        
        return wrapper_gate

    models = []
    num_models = 2

    for _ in range(num_models):
        m = FMoETransformerMLP(is_benchmark=True, num_expert=n_expert, d_model=d_model, d_hidden=d_model*2, world_size=world_size, gate=file_gate(filename, layer_idx)).cuda()
        models.append(m)        

    history = np.array(load_gate_history(table_prefix+f"_iter{iter - history_lat}", layer_idx))

    with torch.no_grad():
        for e in range(n_expert):
            for i in range(1, num_models):
                models[i].experts[e].htoh4.weight.copy_(models[0].experts[e].htoh4.weight)
                models[i].experts[e].htoh4.bias.copy_(models[0].experts[e].htoh4.bias)
            
        for i in range(1, num_models):
            models[i].gate.gate_top_k_idx_list = models[0].gate.gate_top_k_idx_list

    models_fp16 = []
    for i in range(num_models):
        models_fp16.append(module.Float16Module(models[i]))

    n_run = 4
    
    x = []

    x0 = torch.rand(seq_len * batch_size, d_model, dtype=torch.float16).cuda()
    x0.requires_grad = True
    x.append(x0)

    for i in range(1, num_models):
        xi = x0.data.clone()
        xi.requires_grad = True
        x.append(xi)

    global smartmoe_last_mapping

    for round in range(n_run):
        # model 0: FastMoE
        torch.cuda.synchronize()
        t0_begin = time.time()

        y0 = models_fp16[0](x[0], force_no_shadow=True)
        
        torch.cuda.synchronize()
        t0_end = time.time()

        loss = y0.sum()

        torch.cuda.synchronize()
        t0_bwd_begin = time.time()

        loss.backward()

        torch.cuda.synchronize() 
        t0_bwd_end = time.time()
        
        # model 1: SmartMoE
        torch.cuda.synchronize()
        t1_begin = time.time()

        y1 = models_fp16[1](x[1], benchmark_history_gate=history if round == 0 and enable_update else None, benchmark_method='Greedy', benchmark_last_mapping=smartmoe_last_mapping[layer_idx])

        torch.cuda.synchronize()
        t1_end = time.time()
        smartmoe_last_mapping[layer_idx] = models_fp16[1].module.expert_mapping

        loss = y1.sum()

        torch.cuda.synchronize()
        t1_bwd_begin = time.time()

        loss.backward()

        torch.cuda.synchronize() 
        t1_bwd_end = time.time()

        t0 = t0_end - t0_begin + t0_bwd_end - t0_bwd_begin
        t1 = t1_end - t1_begin + t1_bwd_end - t1_bwd_begin

        if rank == 0:
            print(f"{spec}: round {round} FastMoE={t0:.3f} SmartMoE={t1:.3f}", flush=True)


if __name__ == '__main__':
    table_prefix = sys.argv[1]
    batch_size = int(sys.argv[2])
    d_model = int(sys.argv[3])
    history_lat = int(sys.argv[4])
    update_freq = int(sys.argv[5])

    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tot_expert = 32
    assert tot_expert % world_size == 0

    test_iters = []
    for idx in range(1, 3000):
        if idx % 10 == 1 and idx - history_lat > 0:
            test_iters.append(idx)

    test_layers = [6]
    for layer in test_layers:
        origin_mapping = [idx for idx in range(tot_expert)]
        smartmoe_last_mapping[layer] = origin_mapping

    for iter in test_iters:
        for layer_idx in test_layers:
            
            enable_update = iter % update_freq == 1

            filename = table_prefix + f"_iter{iter}"

            seq_len = 1024
            num_expert = 32 // world_size
            spec = f"test d_model {d_model} seq_len {seq_len} batch_size {batch_size} num_expert {num_expert} world_size {world_size} iter {iter} layer {layer_idx}"
            _test_smart_exchange(spec, table_prefix, iter, enable_update, filename, layer_idx, history_lat, d_model, seq_len, batch_size, num_expert)
