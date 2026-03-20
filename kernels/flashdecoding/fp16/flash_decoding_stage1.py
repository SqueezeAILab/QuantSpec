import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q, K, V, sm_scale,
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_ks,
    stride_vbs, stride_vh, stride_vd,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_seq_start_index = seq_start_block * BLOCK_SEQ
    cur_seq_end_index = (seq_start_block + 1) * BLOCK_SEQ

    # query states offset
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    off_kbh = cur_batch * stride_kbs + cur_head * stride_kh
    off_vbh = cur_batch * stride_vbs + cur_head * stride_vh
    
    # How many small blocks within a large block
    block_n_size = tl.cdiv(BLOCK_SEQ, BLOCK_N)
    
    # key states and value states offset
    offs_n = cur_seq_start_index + tl.arange(0, BLOCK_N)
    
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        off_k = off_kbh + offs_n_new[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        # tl.static_print(q, k, q[None, :] * k, att_value) # (128), (64, 128), (64, 128), (64) 

        off_v = off_vbh + offs_d[:, None] * stride_vd + offs_n_new[None, :]
        v = tl.load(V + off_v)
        
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        # tl.static_print(v, att_value, cur_max_logic, max_logic, new_max_logic) # (128, 64), (64), 1, 1, 1

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[None, :] * v, axis=1)

        # tl.static_print(exp_logic, logic_scale, acc) # (64), 1, (128)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

        # tl.static_print(sum_exp, max_logic) # 1, 1

    off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
    tl.store(Mid_O + off_mid_o, acc / sum_exp)
    tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))

@torch.no_grad()
def flash_decode_stage1(q, k, v, mid_out, mid_out_logsumexp, qcache_len, block_seq):
    """
    q : torch.Tensor
        The query tensor of shape (batch_size, num_heads, head_dim). It represents 
        the query for the current token at the specific decoding step.
    
    cache_k : torch.Tensor
        The cached key tensor of shape (batch_size, num_heads, seq_length, head_dim). 

    cache_v : torch.Tensor
        The cached value tensor of shape (batch_size, num_heads, head_dim, seq_length). 
        It is transposed to meet the requirement of decoding
    
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
    BLOCK_N = 64
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256, 512}
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, head_num = q.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(qcache_len, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]

    # import IPython
    # IPython.embed()

    _fwd_kernel_flash_decode_stage1[grid](
        q, k, v, sm_scale,
        mid_out,
        mid_out_logsumexp,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )
    return