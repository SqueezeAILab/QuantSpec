import triton
import triton.language as tl
import random
import numpy as np
import torch

from .quant_pack_int4toint32 import _minmax_along_penultimate_dim

@triton.jit
def _int4toint8_pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	x_stride_b: tl.constexpr, 
	code_stride_b: tl.constexpr, 
	group_size: tl.constexpr,
	code_size: tl.constexpr,
	D: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	pid_b = tl.program_id(axis=0) # batch * num_head * num_group
	pid_d = tl.program_id(axis=1) # hidden size for k, sequence length for v

	x_tile_ptr = tl.make_block_ptr(
		base=intensor_ptr + pid_b * x_stride_b,
		shape=(group_size, D), 
		strides=(D, 1),
        offsets=(0, pid_d * BLOCK_SIZE_N * feat_per_int),
		block_shape=(group_size, BLOCK_SIZE_N * feat_per_int),
        order=(1, 0)
	)

	packed = tl.zeros((group_size, BLOCK_SIZE_N,), dtype=tl.int8)

	element = tl.load(x_tile_ptr)
	element = tl.reshape(element, (group_size, BLOCK_SIZE_N, 2))
	element_low, element_high = tl.split(element)
	element_high = element_high << bits

	# Combine the value using bitwise OR
	packed = element_high | element_low
	packed = packed.to(tl.int8)

	code_tile_ptr = tl.make_block_ptr(
		base=code_ptr + pid_b * code_stride_b,
		shape=(group_size, code_size), 
		strides=(code_size, 1),
        offsets=(0, pid_d * BLOCK_SIZE_N),
		block_shape=(group_size, BLOCK_SIZE_N),
        order=(1, 0)
	)

	tl.store(code_tile_ptr, packed)


# ChatGPT told me shape[-2] is the penultimate dim
def triton_int4toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * num_groups, group_size, D)
	scale_mn_shape = (B, nh, num_groups, D)
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * num_groups, D), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * num_groups, D), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 64
	grid = lambda meta: (data.shape[0], triton.cdiv(D, BLOCK_SIZE_N))

	_minmax_along_penultimate_dim[grid](data, mn, mx, data.stride(0),
							 group_size, D,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)

	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(1)
	data.div_(scale.unsqueeze(1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int8)
	feat_per_int = 8 // bit
	code_size = D // feat_per_int
	packshape = (B * nh * num_groups, group_size, code_size,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int8)
	grid = lambda meta: (data.shape[0], triton.cdiv(code_size, BLOCK_SIZE_N))

	_int4toint8_pack_along_last_dim[grid](bit, data, code, data.stride(0), code.stride(0),
								group_size, code_size, D, feat_per_int,
								BLOCK_SIZE_N=BLOCK_SIZE_N,
								num_warps=8)

	return code.view(B, nh, num_groups * group_size, code_size), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
