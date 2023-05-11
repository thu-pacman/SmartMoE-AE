r"""
FMoE core layer
"""
import tree
import os
import torch
import torch.nn as nn

from .functions import prepare_forward, ensure_comm
from .functions import MOEScatter, MOEGather
from .functions import AllGather, Slice
from .gates import NaiveGate, FileGate

from .fastermoe.config import switch_from_env
from .utils import get_torch_default_comm
from .fastermoe.expert_utils import get_expert_param_size, get_expert_param_dtype, get_expert_params, set_params

from .smartmoe import generate_mapping_from_history

def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )

    x = tree.map_structure(scatter_func, inp)

    x = expert_fn(x, fwd_expert_count)

    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )

    outp = tree.map_structure(gather_func, x)
    return outp


fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        is_benchmark=False,
        args=None
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.tot_expert = num_expert * world_size

        if not is_benchmark:
            from .megatron.distributed import get_gate_group_fn,get_expert_ep_group_fn
            from .fastermoe.config import switch_from_env
            
            self.dynamic_placement = args.dynamic_placement
            self.dynamic_freq = args.dynamic_freq
            
            self.gate_group = get_gate_group_fn()
            self.ep_group = get_expert_ep_group_fn()
            self.rank = torch.distributed.get_rank(group=self.ep_group)
            self.is_ep_global = False
        else:
            self.ep_group = get_torch_default_comm()
            self.rank = torch.distributed.get_rank()
            self.is_ep_global = True

        # expert i is place on expert_mapping[i]
        self.expert_mapping = [idx for idx in range(self.tot_expert)]

        self.gate_history = torch.zeros(self.tot_expert).cuda()
        self.history_cnt = 0
        self.iter_cnt = 0

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        
        self.is_benchmark = is_benchmark

        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

        self.ex = self.expert_fn 

        global fmoe_faster_schedule
        if fmoe_faster_schedule :
            self.ex = self.expert_fn_single

        global _fmoe_general_global_forward
        self.fwd_fn = _fmoe_general_global_forward

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)
    
    def expert_fn_single(self, inp, fwd_expert_count, idx):
        r"""
        forward single expert for smart scheduling.
        """
        assert not self.experts_fused, "should not use fused experts"
        output = self.experts[idx](inp, fwd_expert_count)
        return output

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def update_gate_history(self, gate_top_k_idx):
        with torch.no_grad():
            buffer = torch.zeros(self.tot_expert).cuda()
            valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
            buffer.scatter_add_(0,
                valid_idx.view(-1),
                torch.ones_like(valid_idx.view(-1), dtype=torch.float)
            )

            self.gate_history = (self.gate_history * self.history_cnt + buffer) / (self.history_cnt + 1)
            self.history_cnt += 1

    def is_same_mapping(self, new_mapping):
        with torch.no_grad():
            for idx in range(self.tot_expert):
                if new_mapping[idx] != self.expert_mapping[idx]:
                    return False
            return True

    def update_expert_mapping(self, new_mapping):
        with torch.no_grad():
            assert isinstance(self.experts, nn.ModuleList)
            self.gate_history = torch.zeros(self.tot_expert).cuda()
            self.history_cnt = 0

            if self.is_same_mapping(new_mapping):
                # nothing need to do
                return
            if self.rank == 0:
                print("[INFO] mapping updated!", flush=True)
            
            my_send = []
            my_recv = []
            params_size = get_expert_param_size(self.experts, 0)
            params_type = get_expert_param_dtype(self.experts, 0)
            send_buffer = torch.zeros(params_size, dtype=params_type).cuda()
            recv_list = []
            for idx in range(self.num_expert*self.world_size):
                if self.expert_mapping[idx] == new_mapping[idx]:
                    continue
                if self.expert_mapping[idx] // self.num_expert == self.rank:
                    local_idx = self.expert_mapping[idx] % self.num_expert
                    to_rank = new_mapping[idx] // self.num_expert
                    to_local_idx = new_mapping[idx] % self.num_expert
                    my_send.append((local_idx,to_rank,to_local_idx))
                    get_expert_params(self.experts, send_buffer, local_idx)
                    if to_rank != self.rank:
                        if self.is_ep_global:
                            global_to_rank = to_rank
                        else:
                            global_to_rank = torch.distributed.distributed_c10d._get_global_rank(self.ep_group, to_rank)
                        torch.distributed.isend(send_buffer, global_to_rank).wait()

                if new_mapping[idx] // self.num_expert == self.rank:
                    from_local_idx = self.expert_mapping[idx] % self.num_expert
                    from_rank = self.expert_mapping[idx] // self.num_expert
                    local_idx = new_mapping[idx] % self.num_expert
                    my_recv.append((local_idx, from_rank, from_local_idx))
                    recv_buffer = torch.zeros_like(send_buffer)
                    if from_rank != self.rank:
                        if self.is_ep_global:
                            global_from_rank = from_rank
                        else:
                            global_from_rank = torch.distributed.distributed_c10d._get_global_rank(self.ep_group, from_rank)
                        torch.distributed.irecv(recv_buffer, global_from_rank).wait()
                    else:
                        recv_buffer = send_buffer.data.clone()
                    recv_list.append(recv_buffer)

            for recv_idx in range(len(recv_list)):
                set_params(self.experts, recv_list[recv_idx], my_recv[recv_idx][0])

            self.expert_mapping = new_mapping

    def forward(self, moe_inp, is_first_micro_batch=True, is_last_micro_batch=True, benchmark_history_gate=None, benchmark_method=None, force_no_shadow=False, benchmark_last_mapping=None):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        if self.is_benchmark:
            if benchmark_last_mapping is not None:
                self.expert_mapping = benchmark_last_mapping
            if benchmark_history_gate is not None:
                new_mapping = generate_mapping_from_history(self.num_expert, self.world_size, self.expert_mapping, benchmark_history_gate, method=benchmark_method)
                self.update_expert_mapping(new_mapping)
                if self.rank == 0:
                    print(new_mapping)
                
        else:
            if self.dynamic_placement and is_first_micro_batch and self.iter_cnt % self.dynamic_freq == self.dynamic_freq - 1:
                torch.distributed.all_reduce(self.gate_history, group=self.gate_group)

                new_mapping = generate_mapping_from_history(self.num_expert, self.world_size, self.expert_mapping, self.gate_history.view(-1).cpu().detach().numpy())
                self.update_expert_mapping(new_mapping)
        
            self.iter_cnt += is_first_micro_batch                

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)

        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if (not self.is_benchmark) and self.dynamic_placement and is_first_micro_batch and self.iter_cnt % self.dynamic_freq >= self.dynamic_freq - 2:
            self.update_gate_history(gate_top_k_idx)

        if self.is_benchmark:
            gate_top_k_idx = gate_top_k_idx.cpu().apply_(lambda x: self.expert_mapping[x]).cuda()
        else:
            if self.dynamic_placement:
                gate_top_k_idx = gate_top_k_idx.cpu().apply_(lambda x: self.expert_mapping[x]).cuda()

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = self.fwd_fn(
            moe_inp, gate_top_k_idx, self.ex,
            self.num_expert, self.world_size,
            experts=self.experts,
            is_first_micro_batch=is_first_micro_batch,
            is_last_micro_batch=is_last_micro_batch,
            force_no_shadow=force_no_shadow
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:
            
            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        return moe_outp
