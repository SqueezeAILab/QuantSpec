import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv_verify_upperlower_flash_decode_stage1(
    Q, # query
    Quant_K_Upper, Quant_K_Lower, Scale_K, Min_K,
    Quant_V_Upper, Quant_V_Lower, Scale_V, Min_V,
    sm_scale,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_qbs, stride_qh, stride_qtoken, stride_qd,
    stride_quant_kbs, stride_quant_kh, stride_quant_ks, # Batch Size, Head, Sequence
    stride_scale_kbs, stride_scale_kh, stride_scale_ks, # Scale and Min should share this stride
    stride_quant_vbs, stride_quant_vh, stride_quant_vd, # Batch Size, Head, head Dim
    stride_scale_vbs, stride_scale_vh, stride_scale_vd, # Scale and Min should share this stride
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_otoken, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_mid_o_etoken,
    kbit: tl.constexpr, vbit: tl.constexpr,
    group_size: tl.constexpr,
    elem_per_int: tl.constexpr,
    gqa_group_size,
    QCACHE_LEN, # Not tl.constexpt since we need CUDA Graph so we can not use .item()
    VERIFY_LEN: tl.constexpr,
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
    
    QCACHE_LEN_INT = tl.load(QCACHE_LEN)
    if seq_start_block >= tl.cdiv(QCACHE_LEN_INT, BLOCK_SEQ): # Move this from triton's grid to the kernel to make it compatible with CUDA Graph & torch.compile
        return
    
    # cur_kv_head = cur_head // gqa_group_size

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
    off_q_token0 = cur_batch * stride_qbs + cur_head * stride_qh + 0 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 1:
        off_q_token1 = cur_batch * stride_qbs + cur_head * stride_qh + 1 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 2:
        off_q_token2 = cur_batch * stride_qbs + cur_head * stride_qh + 2 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 3:
        off_q_token3 = cur_batch * stride_qbs + cur_head * stride_qh + 3 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 4:
        off_q_token4 = cur_batch * stride_qbs + cur_head * stride_qh + 4 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 5:
        off_q_token5 = cur_batch * stride_qbs + cur_head * stride_qh + 5 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 6:
        off_q_token6 = cur_batch * stride_qbs + cur_head * stride_qh + 6 * stride_qtoken + offs_q_d
    if VERIFY_LEN > 7:
        off_q_token7 = cur_batch * stride_qbs + cur_head * stride_qh + 7 * stride_qtoken + offs_q_d
        
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
    
    q_token0 = tl.load(Q + off_q_token0)
    if VERIFY_LEN > 1:
        q_token1 = tl.load(Q + off_q_token1)
    if VERIFY_LEN > 2:
        q_token2 = tl.load(Q + off_q_token2)
    if VERIFY_LEN > 3:
        q_token3 = tl.load(Q + off_q_token3)
    if VERIFY_LEN > 4:
        q_token4 = tl.load(Q + off_q_token4)
    if VERIFY_LEN > 5:
        q_token5 = tl.load(Q + off_q_token5)
    if VERIFY_LEN > 6:
        q_token6 = tl.load(Q + off_q_token6)
    if VERIFY_LEN > 7:
        q_token7 = tl.load(Q + off_q_token7)

    sum_exp_token0 = 0.0
    max_logic_token0 = -float("inf")
    acc_token0 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 1:
        sum_exp_token1 = 0.0
        max_logic_token1 = -float("inf")
        acc_token1 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 2:
        sum_exp_token2 = 0.0
        max_logic_token2 = -float("inf")
        acc_token2 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 3:
        sum_exp_token3 = 0.0
        max_logic_token3 = -float("inf")
        acc_token3 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 4:
        sum_exp_token4 = 0.0
        max_logic_token4 = -float("inf")
        acc_token4 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 5:
        sum_exp_token5 = 0.0
        max_logic_token5 = -float("inf")
        acc_token5 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 6:
        sum_exp_token6 = 0.0
        max_logic_token6 = -float("inf")
        acc_token6 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    if VERIFY_LEN > 7:
        sum_exp_token7 = 0.0
        max_logic_token7 = -float("inf")
        acc_token7 = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
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

        att_value_token0 = tl.sum(q_token0[None, :] * k, 1)
        att_value_token0 *= sm_scale
        if VERIFY_LEN > 1:
            att_value_token1 = tl.sum(q_token1[None, :] * k, 1)
            att_value_token1 *= sm_scale
        if VERIFY_LEN > 2:
            att_value_token2 = tl.sum(q_token2[None, :] * k, 1)
            att_value_token2 *= sm_scale
        if VERIFY_LEN > 3:
            att_value_token3 = tl.sum(q_token3[None, :] * k, 1)
            att_value_token3 *= sm_scale
        if VERIFY_LEN > 4:
            att_value_token4 = tl.sum(q_token4[None, :] * k, 1)
            att_value_token4 *= sm_scale
        if VERIFY_LEN > 5:
            att_value_token5 = tl.sum(q_token5[None, :] * k, 1)
            att_value_token5 *= sm_scale
        if VERIFY_LEN > 6:
            att_value_token6 = tl.sum(q_token6[None, :] * k, 1)
            att_value_token6 *= sm_scale
        if VERIFY_LEN > 7:
            att_value_token7 = tl.sum(q_token7[None, :] * k, 1)
            att_value_token7 *= sm_scale

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
        
        cur_max_logic_token0 = tl.max(att_value_token0, axis=0)
        new_max_logic_token0 = tl.maximum(cur_max_logic_token0, max_logic_token0)
        if VERIFY_LEN > 1:
            cur_max_logic_token1 = tl.max(att_value_token1, axis=0)
            new_max_logic_token1 = tl.maximum(cur_max_logic_token1, max_logic_token1)
        if VERIFY_LEN > 2:
            cur_max_logic_token2 = tl.max(att_value_token2, axis=0)
            new_max_logic_token2 = tl.maximum(cur_max_logic_token2, max_logic_token2)
        if VERIFY_LEN > 3:
            cur_max_logic_token3 = tl.max(att_value_token3, axis=0)
            new_max_logic_token3 = tl.maximum(cur_max_logic_token3, max_logic_token3)
        if VERIFY_LEN > 4:
            cur_max_logic_token4 = tl.max(att_value_token4, axis=0)
            new_max_logic_token4 = tl.maximum(cur_max_logic_token4, max_logic_token4)
        if VERIFY_LEN > 5:
            cur_max_logic_token5 = tl.max(att_value_token5, axis=0)
            new_max_logic_token5 = tl.maximum(cur_max_logic_token5, max_logic_token5)
        if VERIFY_LEN > 6:
            cur_max_logic_token6 = tl.max(att_value_token6, axis=0)
            new_max_logic_token6 = tl.maximum(cur_max_logic_token6, max_logic_token6)
        if VERIFY_LEN > 7:
            cur_max_logic_token7 = tl.max(att_value_token7, axis=0)
            new_max_logic_token7 = tl.maximum(cur_max_logic_token7, max_logic_token7)

        exp_logic_token0 = tl.exp(att_value_token0 - new_max_logic_token0)
        logic_scale_token0 = tl.exp(max_logic_token0 - new_max_logic_token0)
        acc_token0 *= logic_scale_token0
        acc_token0 += tl.sum(exp_logic_token0[None, :] * v, axis=1)
        if VERIFY_LEN > 1:
            exp_logic_token1 = tl.exp(att_value_token1 - new_max_logic_token1)
            logic_scale_token1 = tl.exp(max_logic_token1 - new_max_logic_token1)
            acc_token1 *= logic_scale_token1
            acc_token1 += tl.sum(exp_logic_token1[None, :] * v, axis=1)
        if VERIFY_LEN > 2:
            exp_logic_token2 = tl.exp(att_value_token2 - new_max_logic_token2)
            logic_scale_token2 = tl.exp(max_logic_token2 - new_max_logic_token2)
            acc_token2 *= logic_scale_token2
            acc_token2 += tl.sum(exp_logic_token2[None, :] * v, axis=1)
        if VERIFY_LEN > 3:
            exp_logic_token3 = tl.exp(att_value_token3 - new_max_logic_token3)
            logic_scale_token3 = tl.exp(max_logic_token3 - new_max_logic_token3)
            acc_token3 *= logic_scale_token3
            acc_token3 += tl.sum(exp_logic_token3[None, :] * v, axis=1)
        if VERIFY_LEN > 4:
            exp_logic_token4 = tl.exp(att_value_token4 - new_max_logic_token4)
            logic_scale_token4 = tl.exp(max_logic_token4 - new_max_logic_token4)
            acc_token4 *= logic_scale_token4
            acc_token4 += tl.sum(exp_logic_token4[None, :] * v, axis=1)
        if VERIFY_LEN > 5:
            exp_logic_token5 = tl.exp(att_value_token5 - new_max_logic_token5)
            logic_scale_token5 = tl.exp(max_logic_token5 - new_max_logic_token5)
            acc_token5 *= logic_scale_token5
            acc_token5 += tl.sum(exp_logic_token5[None, :] * v, axis=1)
        if VERIFY_LEN > 6:
            exp_logic_token6 = tl.exp(att_value_token6 - new_max_logic_token6)
            logic_scale_token6 = tl.exp(max_logic_token6 - new_max_logic_token6)
            acc_token6 *= logic_scale_token6
            acc_token6 += tl.sum(exp_logic_token6[None, :] * v, axis=1)
        if VERIFY_LEN > 7:
            exp_logic_token7 = tl.exp(att_value_token7 - new_max_logic_token7)
            logic_scale_token7 = tl.exp(max_logic_token7 - new_max_logic_token7)
            acc_token7 *= logic_scale_token7
            acc_token7 += tl.sum(exp_logic_token7[None, :] * v, axis=1)

        sum_exp_token0 = sum_exp_token0 * logic_scale_token0 + tl.sum(exp_logic_token0, axis=0)
        max_logic_token0 = new_max_logic_token0
        if VERIFY_LEN > 1:
            sum_exp_token1 = sum_exp_token1 * logic_scale_token1 + tl.sum(exp_logic_token1, axis=0)
            max_logic_token1 = new_max_logic_token1
        if VERIFY_LEN > 2:
            sum_exp_token2 = sum_exp_token2 * logic_scale_token2 + tl.sum(exp_logic_token2, axis=0)
            max_logic_token2 = new_max_logic_token2
        if VERIFY_LEN > 3:
            sum_exp_token3 = sum_exp_token3 * logic_scale_token3 + tl.sum(exp_logic_token3, axis=0)
            max_logic_token3 = new_max_logic_token3
        if VERIFY_LEN > 4:
            sum_exp_token4 = sum_exp_token4 * logic_scale_token4 + tl.sum(exp_logic_token4, axis=0)
            max_logic_token4 = new_max_logic_token4
        if VERIFY_LEN > 5:
            sum_exp_token5 = sum_exp_token5 * logic_scale_token5 + tl.sum(exp_logic_token5, axis=0)
            max_logic_token5 = new_max_logic_token5
        if VERIFY_LEN > 6:
            sum_exp_token6 = sum_exp_token6 * logic_scale_token6 + tl.sum(exp_logic_token6, axis=0)
            max_logic_token6 = new_max_logic_token6
        if VERIFY_LEN > 7:
            sum_exp_token7 = sum_exp_token7 * logic_scale_token7 + tl.sum(exp_logic_token7, axis=0)
            max_logic_token7 = new_max_logic_token7

    off_mid_o_token0 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 0 * stride_mid_otoken + offs_q_d # offset_d is renamed
    off_mid_o_logexpsum_token0 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 0
    tl.store(Mid_O + off_mid_o_token0, acc_token0 / sum_exp_token0)
    tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token0, max_logic_token0 + tl.log(sum_exp_token0))
    if VERIFY_LEN > 1:
        off_mid_o_token1 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 1 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token1 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 1
        tl.store(Mid_O + off_mid_o_token1, acc_token1 / sum_exp_token1)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token1, max_logic_token1 + tl.log(sum_exp_token1))
    if VERIFY_LEN > 2:
        off_mid_o_token2 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 2 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token2 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 2
        tl.store(Mid_O + off_mid_o_token2, acc_token2 / sum_exp_token2)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token2, max_logic_token2 + tl.log(sum_exp_token2))
    if VERIFY_LEN > 3:
        off_mid_o_token3 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 3 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token3 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 3
        tl.store(Mid_O + off_mid_o_token3, acc_token3 / sum_exp_token3)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token3, max_logic_token3 + tl.log(sum_exp_token3))
    if VERIFY_LEN > 4:
        off_mid_o_token4 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 4 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token4 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 4
        tl.store(Mid_O + off_mid_o_token4, acc_token4 / sum_exp_token4)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token4, max_logic_token4 + tl.log(sum_exp_token4))
    if VERIFY_LEN > 5:
        off_mid_o_token5 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 5 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token5 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 5
        tl.store(Mid_O + off_mid_o_token5, acc_token5 / sum_exp_token5)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token5, max_logic_token5 + tl.log(sum_exp_token5))
    if VERIFY_LEN > 6:
        off_mid_o_token6 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 6 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token6 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 6
        tl.store(Mid_O + off_mid_o_token6, acc_token6 / sum_exp_token6)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token6, max_logic_token6 + tl.log(sum_exp_token6))
    if VERIFY_LEN > 7:
        off_mid_o_token7 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + 7 * stride_mid_otoken + offs_q_d # offset_d is renamed
        off_mid_o_logexpsum_token7 = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block * stride_mid_o_es + 7
        tl.store(Mid_O + off_mid_o_token7, acc_token7 / sum_exp_token7)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum_token7, max_logic_token7 + tl.log(sum_exp_token7))

@torch.no_grad()
def int8kv_verify_upperlower_flash_decode_stage1(
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
    batch, head_num, verify_len = q.shape[0], q.shape[1], q.shape[2]
    grid = (batch, head_num, triton.cdiv(max_seq_length, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // cache_quant_k_upper.shape[1]

    asm = _fwd_kernel_int8kv_verify_upperlower_flash_decode_stage1[grid](
        q, 
        cache_quant_k_upper, cache_quant_k_lower, cache_scale_k, cache_min_k, 
        cache_quant_v_upper, cache_quant_v_lower, cache_scale_v, cache_min_v,
        sm_scale,
        mid_out,
        mid_out_logsumexp,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cache_quant_k_upper.stride(0), cache_quant_k_upper.stride(1), cache_quant_k_upper.stride(2),
        cache_scale_k.stride(0), cache_scale_k.stride(1), cache_scale_k.stride(2), # Scale and Min should share this stride since they are of same shape
        cache_quant_v_upper.stride(0), cache_quant_v_upper.stride(1), cache_quant_v_upper.stride(2),
        cache_scale_v.stride(0), cache_scale_v.stride(1), cache_scale_v.stride(2), # Scale and Min should share this stride since they are of same shape
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out.stride(4),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2), mid_out_logsumexp.stride(3),
        kbit, vbit,
        group_size,
        elem_per_int,
        gqa_group_size,
        QCACHE_LEN=qcache_len,
        VERIFY_LEN=verify_len,
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