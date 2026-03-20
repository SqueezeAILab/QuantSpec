import math
import time
import torch
import torch.nn as nn

from .int4kv_int32pack_flash_decoding_stage1 import int4kv_int32pack_flash_decode_stage1
from .int4kv_int32pack_flash_decoding_stage2 import int4kv_int32pack_flash_decode_stage2

def token_decode_attention_int4kv_int32pack_flash_decoding(
    q, cache_quant_k, cache_scale_k, cache_min_k, cache_quant_v, cache_scale_v, cache_min_v, \
    kbit, vbit, group_size, out=None, alloc_tensor_func=torch.zeros,
    qcache_len=0
):
    """
    q : torch.Tensor
        The query tensor of shape (batch_size, num_heads, 1, head_dim). It represents 
        the query for the current token at the specific decoding step.
    
    cache_quant_k : torch.Tensor
        The quantized cached key tensor of shape (batch_size, num_heads, seq_length // elem_per_int, head_dim). 
    
    cache_scale_k : torch.Tensor
        The scale factor for quantized cached key tensor of shape (batch_size, num_heads, seq_length // group_size, head_dim). 
    
    cache_min_k : torch.Tensor
        The zero point for quantized cached key tensor of shape (batch_size, num_heads, seq_length // group_size, head_dim). 

    cache_quant_v : torch.Tensor
        The cached value tensor of shape (batch_size, num_heads, head_dim, seq_length // elem_per_int). 
        It is transposed to meet the requirement of decoding.
        
    cache_scale_v : torch.Tensor
        The scale factor for quantized cached value tensor of shape (batch_size, num_heads, head_dim, seq_length // group_size). 
    
    cache_min_v : torch.Tensor
        The zero point for quantized cached value tensor of shape (batch_size, num_heads, head_dim, seq_length // group_size). 

    out : torch.Tensor, optional
        Output tensor to store the result of shape (batch_size, num_heads, 1, head_dim).

    --------
    kbit: int
        Key cache's bit width. Options: [2, 4]

    vbit: int
        Value cache's bit width. Options: [2, 4]

    group_size: int
        How many elements should be quantized as a group. Options: [32, 64, 128]
    
    elem_per_int: int
        How many quantized elements will be packed together into a int32 element.
    """
    # If quantization group size is 32 and batch size = 1, This should be set to 256
    BLOCK_SEQ = 128
    assert kbit == vbit, "We only support kbit == vbit in [2, 4]"
    batch_size = q.shape[0]
    q_head_num, head_dim = q.shape[1], q.shape[-1]
    qcache_len = cache_scale_v.shape[-1]

    elem_per_int = 32 // kbit # 32 since we use int32 to pack the quantized elements
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    o_tensor = alloc_tensor_func(tuple(q.shape), dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, qcache_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device=q.device
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, qcache_len // BLOCK_SEQ + 1], dtype=torch.float32, device=q.device
    )

    int4kv_int32pack_flash_decode_stage1(
        q.view(calcu_shape1),
        cache_quant_k, 
        cache_scale_k, 
        cache_min_k, 
        cache_quant_v, 
        cache_scale_v, 
        cache_min_v,
        mid_o,
        mid_o_logexpsum,
        qcache_len,
        BLOCK_SEQ,
        kbit, 
        vbit, 
        group_size,
        elem_per_int
    )
    int4kv_int32pack_flash_decode_stage2(mid_o, mid_o_logexpsum, o_tensor.view(calcu_shape1), qcache_len, BLOCK_SEQ)
    return o_tensor
