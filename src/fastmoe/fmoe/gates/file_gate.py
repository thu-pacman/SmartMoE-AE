r"""
Naive gate
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import random

random.seed(19981118)

class FileGate(BaseGate):
    r"""
    expert distribution is provided by user.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, rank=None, tensor_type=torch.float16, inp_shape=None, filename=None, layer_idx=None):
        super().__init__(num_expert, world_size)
        self.top_k = top_k

        if rank is None:
            from megatron.mpu import get_expert_ep_group
            rank = torch.distributed.get_rank(group=get_expert_ep_group())

        if inp_shape is None:
            inp_shape = 1024 * 2

        if filename is None:
            filename = os.getenv('TABLE_FILE')
        
        if layer_idx is None:
            layer_idx = int(os.getenv('LAYER_IDX'))

        filename += "_rank{}.log".format(rank)

        self.gate_top_k_idx_list = []
        self.fwd_cnt = 0

        with open(filename, "r") as f:
            lines = f.readlines()
            line = lines[layer_idx]
            idx, table = line.split(':')
            microbatch_tables = json.loads(table)

            base_cnt = sum(microbatch_tables[0])
            assert inp_shape * top_k % base_cnt == 0, f"inp shape mismatch {inp_shape} {top_k} {base_cnt}"
            rate = inp_shape * top_k // base_cnt

            for table in microbatch_tables:
                e_list = []
                for e_idx, e_cnt in enumerate(table):
                    e_list += [int(e_idx) for _ in range(int(e_cnt*rate))]
                    assert e_idx >= 0 and e_idx < self.tot_expert
                assert len(e_list) == inp_shape * top_k, f"file gate shape mismatch: {len(e_list)} {inp_shape} {top_k}"
                random.shuffle(e_list)
                gate_top_k_idx = torch.tensor(e_list, dtype=torch.int64, device='cuda').view(-1, self.top_k).contiguous()
                self.gate_top_k_idx_list.append(gate_top_k_idx)

            self.gate_top_k_val = torch.ones_like(gate_top_k_idx, dtype=tensor_type)
        
        self.fwd_len = len(self.gate_top_k_idx_list)

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        with torch.no_grad():
            gate_top_k_idx = self.gate_top_k_idx_list[self.fwd_cnt]
            assert inp.shape[0] == gate_top_k_idx.view(-1,self.top_k).shape[0]
            self.fwd_cnt = (self.fwd_cnt + 1) % self.fwd_len
            gate_top_k_val = self.gate_top_k_val
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        
        self.set_loss(torch.zeros(1, requires_grad=True).cuda())

        if return_all_scores:
            assert False, "file gate dont have all scores"
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
