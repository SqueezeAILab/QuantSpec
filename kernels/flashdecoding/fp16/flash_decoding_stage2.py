import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    Mid_O, # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    O, #[batch, head, head_dim]
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    stride_obs, stride_oh, stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_LEN: tl.constexpr):

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    block_n_size = tl.cdiv(MAX_LEN, BLOCK_SEQ)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os) # (128)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n) # 1
        new_max_logic = tl.maximum(tlogic, max_logic) # 1

        old_scale = tl.exp(max_logic - new_max_logic) # 1
        acc *= old_scale # 128

        exp_logic = tl.exp(tlogic - new_max_logic) # 1
        acc += exp_logic * tv

        sum_exp = sum_exp * old_scale + exp_logic # 1
        max_logic = new_max_logic # 1

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, O, qcache_len, block_seq):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128, 256, 512}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _fwd_kernel_flash_decode_stage2[grid](
        mid_out, mid_out_logexpsum, O,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        MAX_LEN=qcache_len,
        num_warps=4,
        num_stages=4,
    )
    return