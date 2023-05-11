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

def load_gate_history(filename, layer_idx, batch_size):
    rank = 0
    filename += "_rank{}.log".format(rank)

    with open(filename, "r") as f:
        lines = f.readlines()
        line = lines[layer_idx]
        _, table = line.split(':')
        microbatch_tables = json.loads(table)

        base_cnt = sum(microbatch_tables[0])
        rate = batch_size // base_cnt

        tot_experts = len(microbatch_tables[0])
        assert tot_experts == 32
        expert_count = [microbatch_tables[0][e] * rate for e in range(tot_experts)]

        return expert_count

if __name__ == '__main__':
    filename = sys.argv[1]
    layer_idx = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    batch_size *= 1024 * 2 # seq_len=1024, top_k=2
