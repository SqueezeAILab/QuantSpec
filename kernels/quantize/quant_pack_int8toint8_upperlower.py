import triton
import triton.language as tl
import random
import numpy as np
import torch

from .quant_pack_int4toint32 import _minmax_along_last_dim, _minmax_along_penultimate_dim
from .quant_pack_int4toint8 import _int4toint8_pack_along_last_dim


# ChatGPT told me shape[-2] is the penultimate dim
def triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
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

	scale = (mx - mn) / (2 ** bit - 1)
	scale_lower = scale / (2 ** bit)
	data_upper = data - mn.unsqueeze(1)
	data_upper.div_(scale.unsqueeze(1))
	data_upper = data_upper.clamp_(0, 2 ** bit - 1).round_().to(torch.int8)
	data_lower = data - (data_upper * scale.unsqueeze(1) + mn.unsqueeze(1))
	data_lower.div_(scale_lower.unsqueeze(1))
	data_lower = data_lower.clamp_(-(2 ** (bit - 1)), 2 ** (bit - 1) - 1).round_().to(torch.int8)

	data_lower = data_lower + 8 # UINT8 to represent, not INT8

	# D = (data * 1e5).to(torch.float32)
	# DU = ((data_upper * scale.unsqueeze(1) + mn.unsqueeze(1)) * 1e5).to(torch.float32)
	# DUL = ((data_upper * scale.unsqueeze(1) + data_lower * scale_lower.unsqueeze(1) + mn.unsqueeze(1)) * 1e5).to(torch.float32)

	feat_per_int = 8 // bit
	code_size = D // feat_per_int
	packshape = (B * nh * num_groups, group_size, code_size,)
	code_upper = torch.zeros(*packshape, device=data.device, dtype=torch.int8)
	code_lower = torch.zeros(*packshape, device=data.device, dtype=torch.int8)
	grid = lambda meta: (data.shape[0], triton.cdiv(code_size, BLOCK_SIZE_N))

	_int4toint8_pack_along_last_dim[grid](bit, data_upper, code_upper, data_upper.stride(0), code_upper.stride(0),
								group_size, code_size, D, feat_per_int,
								BLOCK_SIZE_N=BLOCK_SIZE_N,
								num_warps=8)

	_int4toint8_pack_along_last_dim[grid](bit, data_lower, code_lower, data_lower.stride(0), code_lower.stride(0),
								group_size, code_size, D, feat_per_int,
								BLOCK_SIZE_N=BLOCK_SIZE_N,
								num_warps=8)
	
	return code_upper.view(B, nh, num_groups * group_size, code_size), code_lower.view(B, nh, num_groups * group_size, code_size), \
		scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
