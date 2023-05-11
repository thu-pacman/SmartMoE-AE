r"""
Patching some of Megatron-LM's functions to create an MoE model
"""
import torch

def patch_loss_func(loss_func):
    r"""
    Patch model's loss_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron import get_args
    from .distributed import get_moe_group
    from megatron import get_num_microbatches

    if not get_args().balance_strategy:
        return loss_func

    def loss_func_with_balance_loss(model, output_tensor):
        args = get_args()
        assert args.balance_strategy, "Only use patched loss_func when having balance_strategy."
        assert is_pipeline_last_stage(), "Only call loss_func at pipeline last stage."
        
        output = loss_func(output_tensor)
        
        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.encoder.layers
                if l.mlp.gate.has_loss]

        if hasattr(model.language_model, "decoder"):
            loss_list_decoder = [l.mlp.gate.get_loss(clear=False).view(1)
                    for l in model.language_model.decoder.layers
                    if l.mlp.gate.has_loss]
            loss_list.append(loss_list_decoder)
            
        if len(loss_list) == 0:
            return output

        loss_name = args.balance_strategy + "_loss"
        (loss, state_dict), bal_loss = (
            output,
            torch.cat(loss_list).mean() * args.balance_loss_weight / args.pipeline_model_parallel_size
        )

        averaged_bal_loss = bal_loss

        loss += bal_loss

        state_dict[loss_name] = averaged_bal_loss

        return loss, state_dict

    return loss_func_with_balance_loss

def patch_forward_step(forward_step_func):
    r"""
    Patch model's forward_step_func to support balance loss
    """

    from megatron import get_args
    from functools import partial

    if not get_args().balance_strategy:
        return forward_step_func

    def forward_step_with_balance_loss(data_iterator, model, is_first_micro_batch=True, is_last_micro_batch=True):
        output, loss_func = forward_step_func(data_iterator, model, is_first_micro_batch=is_first_micro_batch, is_last_micro_batch=is_last_micro_batch)
        
        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.encoder.layers
                if l.mlp.gate.has_loss]

        bal_loss = torch.cat(loss_list).mean() * get_args().balance_loss_weight / get_args().pipeline_model_parallel_size
        return output, partial(patch_loss_func(loss_func), model), bal_loss

    return forward_step_with_balance_loss


def patch_model_provider(model_provider, gate=None):
    from megatron import get_args

    def fmoefied_model_provider(pre_process, post_process):
        from .layers import fmoefy
        args = get_args()
        if args.hidden_hidden_size:
            hhs = args.hidden_hidden_size
        else:
            hhs = args.hidden_size * 4
            assert hhs % args.top_k == 0
            hhs = hhs // args.top_k
        distributed_experts = True
        return fmoefy(
            model_provider(pre_process=pre_process, post_process=post_process),
            num_experts=args.num_experts,
            hidden_hidden_size=hhs,
            top_k=args.top_k,
            gate=gate,
            distributed_experts=distributed_experts
        )

    return fmoefied_model_provider
