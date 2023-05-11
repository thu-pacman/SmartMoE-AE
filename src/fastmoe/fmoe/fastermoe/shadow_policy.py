import os
import torch
import torch.distributed as dist


from .config import float_from_env, switch_from_env

def global_policy(local_expert_count, _gec, num_expert, world_size, is_first_micro_batch, is_last_micro_batch):
    r"""
    This is the policy for two-layer MLPs, using the formula in the PPoPP paper.
    A few parameters are used in this policy.
    * `d_model`: feature length of the MLP input and output.
    * `alpha`: the ratio of the MLP's hidden size to `d_model`.
    * `bw_net`: bandwidth of the network (GBps)
    * `bw_mm`: computation throughput of performing GeMM (FLOPs)
    """

    is_print = torch.distributed.get_rank() == 0
    
    bw_net = float_from_env('FMOE_FASTER_GLBPLC_NETBW', 50 * 1e9 / 8)
    bw_bcast_net = float_from_env('FMOE_FASTER_GLBPLC_NETBW_Bcast', 50 * 1e9 / 8)
    bw_mm = float_from_env('FMOE_FASTER_GLBPLC_GPUTP', 11.5e12)
    alpha = float_from_env('FMOE_FASTER_GLBPLC_ALPHA', 2)
    d_model = float_from_env('FMOE_FASTER_GLBPLC_DMODEL', 2048)

    from fmoe.megatron.distributed import get_moe_group
    moe_group = get_moe_group()
    local_expert_count_cuda = local_expert_count.cuda()
    agecs = [torch.empty_like(local_expert_count_cuda) for _ in range(world_size)]
    dist.all_gather(agecs, local_expert_count_cuda, group=moe_group)
    all_global_expert_count = torch.stack(agecs).cpu()

    # TODO: data type other than fp16
    data_size = 2 
   
    fwd_expert_counts = all_global_expert_count.sum(0)
    B_ws, indices = fwd_expert_counts.flatten().sort(0, descending=True)

    alphaH2 = alpha * (d_model ** 2)
    B_w = B_ws[0] - all_global_expert_count[torch.div(indices[0],num_expert, rounding_mode='floor')][indices[0]]
    C_w = B_ws[0]
    assert B_w >= 0

    comm = float('+inf')
    send_feature_time = d_model * data_size / bw_net
    send_model_time = 2 * 2 * alphaH2 * data_size / bw_net * 4
    comp_time = 4 * alphaH2 / bw_mm
    lat_base = 3 * comp_time * C_w + 4 * send_feature_time * B_w

    res = torch.zeros(world_size * num_expert, dtype=torch.bool)
    shadow_time = 0

    lat_last = lat_base
    lat_min = lat_base
    best_shadow = 0

    shadow_cnt = 0
    for i, index in enumerate(indices):
        if i + 1 == indices.numel():
            break
        B_k = B_ws[i + 1] - all_global_expert_count[torch.div(indices[i+1],num_expert,rounding_mode='floor')][indices[i+1]]
        C_k = B_ws[i + 1]
        for j in range(0, i+1):
            C_k += all_global_expert_count[torch.div(indices[i+1],num_expert,rounding_mode='floor')][indices[j]]
        
        shadow_time += send_model_time
        lat_new = 3 * comp_time * C_k + 4 * send_feature_time * B_k + shadow_time

        if lat_new >= lat_last:
            break

        shadow_cnt += 1
        if lat_new < lat_min:
            lat_min = lat_new
            best_shadow = shadow_cnt


    
    for idx in range(best_shadow):
        res[indices[idx]] = True
    
    return res


def no_shadow_policy(_lec, _gec, num_expert, world_size, *args, **kwargs):
    res = torch.zeros(world_size * num_expert, dtype=bool)
    return res


def get_shadow_policy(force_no_shadow, d_model=None):
    if force_no_shadow:
        return no_shadow_policy
    if d_model is not None and 'FMOE_FASTER_GLBPLC_DMODEL' not in os.environ:
        os.environ['FMOE_FASTER_GLBPLC_DMODEL'] = str(d_model)
    if not switch_from_env('FMOE_FASTER_SHADOW_ENABLE'):
        return no_shadow_policy
    return global_policy
