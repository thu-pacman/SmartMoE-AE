r"""
distributed support for Megatron
"""
import torch

from fmoe.distributed import DistributedGroupedDataParallel
from fmoe.fastermoe.config import switch_from_env
from fmoe.utils import get_torch_default_comm

_groups = None

use_megatron = False
if switch_from_env('USE_MEGATRON', False):
    use_megatron = True

def _set_groups(**kwargs):
    global _groups
    _groups = kwargs

def get_moe_group():
    return _groups["moe_group"]

_EXPERT_EP_GROUP = None
_EXPERT_DP_GROUP = None
_EXPERT_EP_GROUP_WORLD_SIZE = None

def get_data_parallel_group_fn():
    global use_megatron
    if use_megatron:
        from megatron import mpu
        return mpu.get_data_parallel_group()
    else:
        return get_torch_default_comm()

def get_gate_group_fn():
    global use_megatron
    if use_megatron:
        from megatron import mpu
        return mpu.get_gate_group()
    else:
        return get_torch_default_comm()

def get_expert_ep_group_fn(args=None):
    global use_megatron
    if use_megatron:
        from megatron import mpu
        return mpu.get_expert_ep_group()
    else:
        global _EXPERT_EP_GROUP
        if _EXPERT_EP_GROUP is None:
            _init(args=args)
        return _EXPERT_EP_GROUP

def get_expert_ep_group_world_size(args=None):
    global use_megatron
    if use_megatron:
        from megatron import mpu
        return mpu.get_expert_ep_group_world_size()
    else:
        global _EXPERT_EP_GROUP_WORLD_SIZE
        if _EXPERT_EP_GROUP_WORLD_SIZE is None:
            _init(args=args)
        return _EXPERT_EP_GROUP_WORLD_SIZE

def get_expert_dp_group_fn(args=None):
    global use_megatron
    if use_megatron:
        from megatron import mpu
        return mpu.get_expert_dp_group()
    else:
        global _EXPERT_DP_GROUP
        if _EXPERT_DP_GROUP is None:
            _init(args=args)
        return _EXPERT_DP_GROUP

def _init(args=None):
    global use_megatron
    if not use_megatron:
        assert args is not None
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        stage_size = world_size
        expert_ep_size = args.expert_ep_size
        expert_dp_size = args.expert_dp_size
        
        assert expert_dp_size * expert_ep_size == stage_size

        global _EXPERT_EP_GROUP_WORLD_SIZE
        _EXPERT_EP_GROUP_WORLD_SIZE = expert_ep_size

        for j in range(0, stage_size, expert_ep_size):
            ep_ranks = range(j, j + expert_ep_size)
            group = torch.distributed.new_group(ep_ranks)
            if rank in ep_ranks:
                global _EXPERT_EP_GROUP
                print("[INFO] {} in EP group {}".format(rank,[v for v in ep_ranks]))
                _EXPERT_EP_GROUP = group
        for j in range(0, expert_ep_size):
            dp_ranks = range(j, stage_size, expert_ep_size)
            group = torch.distributed.new_group(dp_ranks)
            if rank in dp_ranks:
                global _EXPERT_DP_GROUP
                print("[INFO] {} in DP group {}".format(rank,[v for v in dp_ranks]))
                _EXPERT_DP_GROUP = group
    else:
        from megatron import get_args
        args = get_args()

    moe_group = get_expert_ep_group_fn()
    moe_dp_group = get_expert_dp_group_fn()

    _set_groups(
            dp_group=get_data_parallel_group_fn(),
            moe_group=moe_group,
            gate_group=get_gate_group_fn(),
            moe_dp_group=moe_dp_group)


class DistributedDataParallel(DistributedGroupedDataParallel):
    r"""
    A wrapper that is used to replace the DDP module provided by Megatron, which
    is adapted to enable the sophiscated parallel and reduction strategies in
    Fast MoE.
    """

    def __init__(self, module, accumulate_allreduce_grads_in_fp32=False, use_contiguous_buffers_in_ddp=False, args=None):
        assert not accumulate_allreduce_grads_in_fp32, "FastMoE not supports accumulate_allrecude_grads_in_fp32"
        assert not use_contiguous_buffers_in_ddp, "FastMoE not supports use_contiguous_buffers_in_ddp"
        if _groups is None:
            _init(args=args)
        super().__init__(module, **_groups)
    
    def set_input_tensor(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.set_input_tensor(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.state_dict(*args, **kwargs)

    def state_dict_for_save_checkpoint(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.state_dict_for_save_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.load_state_dict(*args, **kwargs)
