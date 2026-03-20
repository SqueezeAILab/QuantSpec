import triton
import triton.language as tl
import random
import numpy as np
import torch

@triton.jit
def _int4toint32_pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)


@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)


@triton.jit
def _int4toint32_pack_along_penultimate_dim(
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
	pid_g = tl.program_id(axis=1) # num of int32 in a group.
	pid_d = tl.program_id(axis=2) # hidden size

	x_tile_ptr = tl.make_block_ptr(
		base=intensor_ptr + pid_b * x_stride_b,
		shape=(group_size, D), 
		strides=(D, 1),
        offsets=(pid_g * feat_per_int, pid_d * BLOCK_SIZE_N),
		block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0)
	)

	packed = tl.zeros((1, BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		element = tl.load(x_tile_ptr)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
		x_tile_ptr = tl.advance(x_tile_ptr, (1, 0))

	code_tile_ptr = tl.make_block_ptr(
		base=code_ptr + pid_b * code_stride_b,
		shape=(code_size, D), 
		strides=(D, 1),
        offsets=(pid_g, pid_d * BLOCK_SIZE_N),
		block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0)
	)

	tl.store(code_tile_ptr, packed)

# ChatGPT told me shape[-2] is the penultimate dim
@triton.jit
def _minmax_along_penultimate_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	x_stride_b: tl.constexpr, 
	group_size: tl.constexpr,
	D: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	pid_b = tl.program_id(axis=0)
	pid_d = tl.program_id(axis=1)

	x_tile_ptr = tl.make_block_ptr(
		base=x_ptr + pid_b * x_stride_b, 
		shape=(group_size, D), 
		strides=(D, 1),
        offsets=(0, pid_d * BLOCK_SIZE_N),
		block_shape=(group_size, BLOCK_SIZE_N),
        order=(1, 0)
	)

	x = tl.load(x_tile_ptr)
	mn_val = tl.min(x, axis=0)
	mx_val = tl.max(x, axis=0)
	
	tl.store(mn_ptr + pid_b * D + pid_d * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), mn_val)
	tl.store(mx_ptr + pid_b * D + pid_d * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), mx_val)

def triton_int4toint32_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	_int4toint32_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
	
# ChatGPT told me shape[-2] is the penultimate dim
def triton_int4toint32_quantize_and_pack_along_penultimate_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0, f"T: {T}, group size: {group_size}"
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
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)

	rqx_data = data * scale.unsqueeze(1) + mn.unsqueeze(1)
	feat_per_int = 32 // bit
	code_size = group_size // feat_per_int
	packshape = (B * nh * num_groups, code_size, D,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (data.shape[0], code_size, triton.cdiv(D, BLOCK_SIZE_N))

	_int4toint32_pack_along_penultimate_dim[grid](bit, data, code, data.stride(0), code.stride(0),
								group_size, code_size, D, feat_per_int,
								BLOCK_SIZE_N=BLOCK_SIZE_N,
								num_warps=8)

	return code.view(B, nh, num_groups * code_size, D), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
	

