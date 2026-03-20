import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv_upperlower_flash_decode_stage1(
    Q, # query
    Quant_K_Upper, Quant_K_Lower, Scale_K, Min_K,
    Quant_V_Upper, Quant_V_Lower, Scale_V, Min_V,
    sm_scale,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_qbs, stride_qh, stride_qd,
    stride_quant_kbs, stride_quant_kh, stride_quant_ks, # Batch Size, Head, Sequence
    stride_scale_kbs, stride_scale_kh, stride_scale_ks, # Scale and Min should share this stride
    stride_quant_vbs, stride_quant_vh, stride_quant_vd, # Batch Size, Head, head Dim
    stride_scale_vbs, stride_scale_vh, stride_scale_vd, # Scale and Min should share this stride
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    kbit: tl.constexpr, vbit: tl.constexpr,
    group_size: tl.constexpr,
    elem_per_int: tl.constexpr,
    gqa_group_size,
    QCACHE_LEN, # Not tl.constexpt since we need CUDA Graph so we can not use .item()
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_SEQ_PER_INT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PER_INT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PER_INT: tl.constexpr,
    precision: tl.constexpr
):
    tl.static_assert(kbit == vbit, "kbit and vbit should be the same")
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    # cur_kv_head = cur_head // gqa_group_size
    
    QCACHE_LEN_INT = tl.load(QCACHE_LEN)
    if seq_start_block >= tl.cdiv(QCACHE_LEN_INT, BLOCK_SEQ): # Move this from triton's grid to the kernel to make it compatible with CUDA Graph & torch.compile
        return
    
    offs_q_d = tl.arange(0, BLOCK_DMODEL) # K do not quantize along the head_dim axis
    offs_k_d = tl.arange(0, BLOCK_DMODEL // elem_per_int) # K do not quantize along the head_dim axis
    offs_k_d_scale = tl.arange(0, BLOCK_DMODEL) # K do not quantize along the head_dim axis
    # Value use this to unpack int2/int4 values
    offs_quant_v_d = tl.arange(0, BLOCK_DMODEL) # V quantize along the head_dim axis, so we need to divide it
    offs_scale_v_d = tl.arange(0, BLOCK_DMODEL // group_size) # V quantize along the head_dim axis, so we need to divide it

    # Key use this
    cur_seq_start_index_quant_k = seq_start_block * BLOCK_SEQ
    cur_seq_start_index_scale_k = seq_start_block * BLOCK_SEQ // group_size
    # Value use this
    cur_seq_start_index_quant_v = seq_start_block * BLOCK_SEQ // elem_per_int
    cur_seq_start_index_scale_v = seq_start_block * BLOCK_SEQ

    # Query states offset
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_q_d
    # Key and Value offset
    off_quant_kbh = cur_batch * stride_quant_kbs + cur_head * stride_quant_kh # Index of current Batch and Head
    off_quant_vbh = cur_batch * stride_quant_vbs + cur_head * stride_quant_vh # Index of current Batch and Head
    # Scale and Offset share the same offset
    off_scale_kbh = cur_batch * stride_scale_kbs + cur_head * stride_scale_kh # Index of current Batch and Head
    off_scale_vbh = cur_batch * stride_scale_vbs + cur_head * stride_scale_vh # Index of current Batch and Head
    
    # How many small blocks within a large block
    block_n_size = tl.cdiv(BLOCK_SEQ, BLOCK_N) # also = tl.cdiv(BLOCK_SEQ_PER_INT, BLOCK_N_PER_INT)
    
    # key states and value states offset
    # Key use this to unpack the int2/int4 value
    offs_n_quant_k = cur_seq_start_index_quant_k + tl.arange(0, BLOCK_N)
    offs_n_scale_k = cur_seq_start_index_scale_k + tl.arange(0, BLOCK_N // group_size)
    # Value use this
    offs_n_quant_v = cur_seq_start_index_quant_v + tl.arange(0, BLOCK_N // elem_per_int)
    offs_n_scale_v = cur_seq_start_index_scale_v + tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # tl.device_print("block_n", block_n_size)
    for start_n in range(0, block_n_size, 1): # Along the sequence length axis
        offs_n_quant_k_new = start_n * BLOCK_N + offs_n_quant_k
        offs_n_scale_k_new = (start_n * BLOCK_N) // group_size + offs_n_scale_k

        off_quant_k = off_quant_kbh + offs_n_quant_k_new[:, None] * stride_quant_ks + offs_k_d[None, :]
        off_scale_k = off_scale_kbh + offs_n_scale_k_new[:, None] * stride_scale_ks + offs_k_d_scale[None, :] # scale and zeropoint share this offset
        quant_k_upper = tl.load(Quant_K_Upper + off_quant_k)
        quant_k_lower = tl.load(Quant_K_Lower + off_quant_k)
        scale_k = tl.load(Scale_K + off_scale_k)
        min_k = tl.load(Min_K + off_scale_k)

        up_k_int8_low = (quant_k_upper & 0xF)
        up_k_int8_high = ((quant_k_upper & 0xF0) >> 4)
        up_k = tl.interleave(up_k_int8_low, up_k_int8_high)

        low_k_int8_low = (quant_k_lower & 0xF)
        low_k_int8_high = ((quant_k_lower & 0xF0) >> 4)
        low_k = tl.interleave(low_k_int8_low, low_k_int8_high)

        low_scale_k = scale_k / (2 ** kbit)

        # tl.static_print(scale_k, min_k)

        # if kbit == 4:
        #     k = (quant_k >> k_shifter[:, None]) & 0xF
        # else:
        #     tl.static_assert(False, "kbit is not 4")
        # k = tl.reshape(k, (BLOCK_N // group_size, group_size, BLOCK_DMODEL))
        # scale_k = tl.reshape(scale_k, (BLOCK_N // group_size, 1, BLOCK_DMODEL))
        # min_k = tl.reshape(min_k, (BLOCK_N // group_size, 1, BLOCK_DMODEL))
        if precision == 8:
            k = (up_k * scale_k + (low_k - 8) * low_scale_k) + min_k
        else:
            k = (up_k * scale_k) + min_k
        # k = tl.reshape(k, (BLOCK_N, BLOCK_DMODEL))

        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        offs_n_v_new = start_n * BLOCK_N // elem_per_int + offs_n_quant_v
        offs_n_scale_v_new = start_n * BLOCK_N + offs_n_scale_v

        off_quant_v = off_quant_vbh + offs_quant_v_d[:, None] * stride_quant_vd + offs_n_v_new[None, :]
        off_scale_v = off_scale_vbh + offs_scale_v_d[:, None] * stride_scale_vd + offs_n_scale_v_new[None, :]
        quant_v_upper = tl.load(Quant_V_Upper + off_quant_v)
        quant_v_lower = tl.load(Quant_V_Lower + off_quant_v)
        scale_v = tl.load(Scale_V + off_scale_v)
        min_v = tl.load(Min_V + off_scale_v)
        
        up_v_int8_low = (quant_v_upper & 0xF)
        up_v_int8_high = ((quant_v_upper & 0xF0) >> 4)
        up_v = tl.interleave(up_v_int8_low, up_v_int8_high)
        
        low_v_int8_low = (quant_v_lower & 0xF)
        low_v_int8_high = ((quant_v_lower & 0xF0) >> 4)
        low_v = tl.interleave(low_v_int8_low, low_v_int8_high)

        low_scale_v = scale_v / (2 ** vbit)

        # tl.static_print(up_v, scale_v, low_v, low_scale_v)

        # v = tl.reshape(v, (BLOCK_DMODEL // group_size, group_size, BLOCK_N))
        # scale_v = tl.reshape(scale_v, (BLOCK_DMODEL // group_size, 1, BLOCK_N))
        # min_v = tl.reshape(min_v, (BLOCK_DMODEL // group_size, 1, BLOCK_N))
        if precision == 8:
            v = (up_v * scale_v + (low_v - 8) * low_scale_v) + min_v
        else:
            v = (up_v * scale_v) + min_v
        # v = tl.reshape(v, (BLOCK_DMODEL, BLOCK_N))
        
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[None, :] * v, axis=1)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_q_d # offset_d is renamed
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
    tl.store(Mid_O + off_mid_o, acc / sum_exp)
    tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))

@torch.no_grad()
def int8kv_upperlower_flash_decode_stage1(
    q,
    cache_quant_k_upper, 
    cache_quant_k_lower, 
    cache_scale_k, 
    cache_min_k, 
    cache_quant_v_upper, 
    cache_quant_v_lower, 
    cache_scale_v, 
    cache_min_v,
    mid_out, 
    mid_out_logsumexp,
    max_seq_length,
    qcache_len, 
    block_seq,
    kbit, 
    vbit, 
    group_size,
    elem_per_int,
    precision
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

    mid_out : torch.Tensor
        Intermediate output tensor to store the intermediate result
        of shape (batch_size, num_heads, seq_length // block_seq + 1, head_dim).
        
    mid_out_logsumexp : torch.Tensor
        Intermediate output tensor to store the intermediate logarithmic value
        of shape (batch_size, num_heads, seq_length // block_seq + 1).

    block_seq: int
        The size of a block when parallelizing the sequence length dim when decoding.
    """
    BLOCK_SEQ = block_seq
    BLOCK_N = 128
    assert BLOCK_SEQ % BLOCK_N == 0

    BLOCK_SEQ_PER_INT = BLOCK_SEQ // elem_per_int
    BLOCK_N_PER_INT = BLOCK_N // elem_per_int

    # shape constraints
    Lq, Lk = q.shape[-1], cache_quant_k_upper.shape[-1] * elem_per_int
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256, 512}
    Lk_PER_INT = Lk // elem_per_int

    sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_seq_length, BLOCK_SEQ))

    gqa_group_size = q.shape[1] // cache_quant_k_upper.shape[1]

    asm = _fwd_kernel_int8kv_upperlower_flash_decode_stage1[grid](
        q, 
        cache_quant_k_upper, cache_quant_k_lower, cache_scale_k, cache_min_k, 
        cache_quant_v_upper, cache_quant_v_lower, cache_scale_v, cache_min_v,
        sm_scale,
        mid_out,
        mid_out_logsumexp,
        q.stride(0), q.stride(1), q.stride(2),
        cache_quant_k_upper.stride(0), cache_quant_k_upper.stride(1), cache_quant_k_upper.stride(2),
        cache_scale_k.stride(0), cache_scale_k.stride(1), cache_scale_k.stride(2), # Scale and Min should share this stride since they are of same shape
        cache_quant_v_upper.stride(0), cache_quant_v_upper.stride(1), cache_quant_v_upper.stride(2),
        cache_scale_v.stride(0), cache_scale_v.stride(1), cache_scale_v.stride(2), # Scale and Min should share this stride since they are of same shape
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        kbit, vbit,
        group_size,
        elem_per_int,
        gqa_group_size,
        QCACHE_LEN=qcache_len,
        BLOCK_SEQ=BLOCK_SEQ, # How many int2/int4 elements a block should load
        BLOCK_SEQ_PER_INT=BLOCK_SEQ_PER_INT, # How many int32 elements a block should load
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PER_INT=Lk_PER_INT,
        BLOCK_N=BLOCK_N, # How many int2/int4 elements a sub-block should load
        BLOCK_N_PER_INT=BLOCK_N_PER_INT, # How many int32 elements a sub-block should load
        precision=precision,
        num_warps=8,
        num_stages=4,
    )

    return