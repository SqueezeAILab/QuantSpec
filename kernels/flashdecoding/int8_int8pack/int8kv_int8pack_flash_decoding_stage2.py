import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_int8kv_int8pack_flash_decode_stage2(
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    O, #[batch, head, head_dim]
    Full_O, # [batch, head, 1, head_dim]
    Full_O_LogExpSum, # [batch, head, 1]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    stride_full_ob, stride_full_oh, stride_full_os, stride_full_od,
    stride_full_o_eb, stride_full_o_eh, stride_full_o_es,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_LEN: tl.constexpr,
    HAVE_FULL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    block_n_size = tl.cdiv(MAX_LEN, BLOCK_SEQ) + 1

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    offs_full_v = cur_batch * stride_full_ob + cur_head * stride_full_oh + offs_d
    offs_full_logic = cur_batch * stride_full_o_eb + cur_head * stride_full_o_eh

    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    if HAVE_FULL:
        tv = tl.load(Full_O + offs_full_v + block_seq_n * stride_full_os)
        tlogic = tl.load(Full_O_LogExpSum + offs_full_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
        
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def int8kv_int8pack_flash_decode_stage2(mid_out, mid_out_logexpsum, O, full_mid_out, full_logexpsum, qcache_len, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128, 256, 512}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    if full_mid_out is not None and full_logexpsum is not None:
        _fwd_kernel_int8kv_int8pack_flash_decode_stage2[grid](
            mid_out, mid_out_logexpsum, O,
            full_mid_out, full_logexpsum,
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
            mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            full_mid_out.stride(0), full_mid_out.stride(1), full_mid_out.stride(2), full_mid_out.stride(3),
            full_logexpsum.stride(0), full_logexpsum.stride(1), full_logexpsum.stride(2),
            BLOCK_SEQ=block_seq,
            BLOCK_DMODEL=Lk,
            MAX_LEN=qcache_len,
            HAVE_FULL=True,
            num_warps=4,
            num_stages=2,
        )
    else:
        _fwd_kernel_int8kv_int8pack_flash_decode_stage2[grid](
            mid_out, mid_out_logexpsum, O,
            mid_out, mid_out_logexpsum, # None, None 
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
            mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), # None, None, None, None
            mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2), # None, None, None, None
            BLOCK_SEQ=block_seq,
            BLOCK_DMODEL=Lk,
            MAX_LEN=qcache_len,
            HAVE_FULL=False,
            num_warps=4,
            num_stages=2,
        )
    return