import triton
import triton.language as tl
import random
import numpy as np
import torch

from .quant_pack_int4toint32 import _minmax_along_penultimate_dim

# ChatGPT told me shape[-2] is the penultimate dim
def triton_int8toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
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
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.uint8)
	feat_per_int = 8 // bit
	code_size = D // feat_per_int
	packshape = (B * nh * num_groups, group_size, code_size,)
	grid = lambda meta: (data.shape[0], triton.cdiv(code_size, BLOCK_SIZE_N))

	return data.view(B, nh, num_groups * group_size, code_size), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
