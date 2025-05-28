# Copyright 2024-2025 The Alibaba Wan Team Authors and ewrfcas. All rights reserved.
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def shard_model(
        model,
        device_id,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        process_group=None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        use_orig_params=False,  # this should only be "True" for inference
        model_type="wan"
):
    model = model.to(torch.float32)
    if model_type == "wan":
        block_list = list(model.blocks)
        if hasattr(model, "controlnet") and hasattr(model.controlnet, "controlnet_blocks"):
            block_list += list(model.controlnet.controlnet_blocks)
    elif model_type == "t5":
        block_list = list(model.encoder.block)
    elif model_type == "clip":
        block_list = list(model.vision_model.encoder.layers)
    else:
        raise NotImplementedError(f"Unknown model type {model_type}")
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in block_list),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=use_orig_params)
    return model