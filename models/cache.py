from numpy import dtype
import torch
from quant_kivi.new_pack import triton_quantize_and_pack_along_last_dim
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from typing import List

import math


class SimQCacheV2(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.batch_size = 1 # TODO: change this to support batch size > 1
        self.max_budget = model.config.max_budget
        self.group_size = model.config.group_size
        self.k_bits = model.config.k_bits
        self.v_bits = model.config.v_bits
        self.residual_length = model.config.residual_length
        self.first_few_fp16_layers = model.config.first_few_fp16_layers

        self.hidden_size = model.config.hidden_size
        self.num_key_value_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers
        self.device = model.device # TODO: change this to support multi-gpu

        # Full cache (residual) and quantized cache
        self.full_key_cache_fp16: List[torch.Tensor] = []
        self.full_value_cache_fp16: List[torch.Tensor] = []
        self.full_key_cache: List[torch.Tensor] = []
        self.full_value_cache: List[torch.Tensor] = []
        self.quantize_key_cache_ubits: List[torch.Tensor] = []
        self.quantize_key_cache_lbits: List[torch.Tensor] = []
        self.trans_quantize_value_cache_ubits: List[torch.Tensor] = []
        self.trans_quantize_value_cache_lbits: List[torch.Tensor] = []

        # Zeropoint and Scale
        self.min_key_cache: List[torch.Tensor] = []
        self.trans_min_value_cache: List[torch.Tensor] = []
        self.scale_key_cache: List[torch.Tensor] = []
        self.trans_scale_value_cache: List[torch.Tensor] = []

        self.dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
        self.qtype = torch.int32

        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        fp16_full_key_cache_shape = (self.batch_size, self.num_key_value_heads, self.max_budget, self.head_dim)
        fp16_full_value_cache_shape = (self.batch_size, self.num_key_value_heads, self.max_budget, self.head_dim)
        full_key_cache_shape = (self.batch_size, self.num_key_value_heads, self.residual_length, self.head_dim) # TODO: this part should be changed to support speculative decoding
        full_value_cache_shape = (self.batch_size, self.num_key_value_heads, self.residual_length, self.head_dim)
        quantize_key_cache_shape = (self.batch_size, self.num_key_value_heads, self.max_budget, self.head_dim)
        trans_quantize_value_cache_shape = (self.batch_size, self.num_key_value_heads, self.head_dim, self.max_budget)

        min_key_cache_shape = (self.batch_size, self.num_key_value_heads, self.max_budget // self.group_size, self.head_dim)
        trans_min_value_cache_shape = (self.batch_size, self.num_key_value_heads, self.head_dim // self.group_size, self.max_budget)
        scale_key_cache_shape = (self.batch_size, self.num_key_value_heads, self.max_budget // self.group_size, self.head_dim)
        trans_scale_value_cache_shape = (self.batch_size, self.num_key_value_heads, self.head_dim // self.group_size, self.max_budget)

        for idx in range(self.layers):
            if idx < self.first_few_fp16_layers:
                self.register_buffer(f"full_key_cache_{idx}", torch.zeros(fp16_full_key_cache_shape, dtype=torch.float16, device=self.device))
                self.register_buffer(f"full_value_cache_{idx}", torch.zeros(fp16_full_value_cache_shape, dtype=torch.float16, device=self.device))
                new_full_layer_key_cache = getattr(self, f"full_key_cache_{idx}")
                new_full_layer_value_cache = getattr(self, f"full_value_cache_{idx}")
                self.full_key_cache_fp16.append(new_full_layer_key_cache)
                self.full_key_cache_fp16.append(new_full_layer_value_cache)
            else:
                self.register_buffer(f"full_key_cache_{idx}", torch.zeros(full_key_cache_shape, dtype=self.dtype, device=self.device))
                self.register_buffer(f"full_value_cache_{idx}", torch.zeros(full_value_cache_shape, dtype=self.dtype, device=self.device))
                self.register_buffer(f"quantize_key_cache_ubits{idx}", torch.zeros(quantize_key_cache_shape, dtype=self.qtype, device=self.device))
                self.register_buffer(f"quantize_key_cache_lbits{idx}", torch.zeros(quantize_key_cache_shape, dtype=self.qtype, device=self.device))
                self.register_buffer(f"trans_quantize_value_cache_ubits{idx}", torch.zeros(trans_quantize_value_cache_shape, dtype=self.qtype, device=self.device))
                self.register_buffer(f"trans_quantize_value_cache_lbits{idx}", torch.zeros(trans_quantize_value_cache_shape, dtype=self.qtype, device=self.device))

                self.register_buffer(f"min_key_cache_{idx}", torch.zeros(min_key_cache_shape, dtype=self.dtype, device=self.device))
                self.register_buffer(f"trans_min_value_cache_{idx}", torch.zeros(trans_min_value_cache_shape, dtype=self.dtype, device=self.device))
                self.register_buffer(f"scale_key_cache_{idx}", torch.zeros(scale_key_cache_shape, dtype=self.dtype, device=self.device))
                self.register_buffer(f"trans_scale_value_cache_{idx}", torch.zeros(trans_scale_value_cache_shape, dtype=self.dtype, device=self.device))
                new_full_layer_key_cache = getattr(self, f"full_key_cache_{idx}")
                new_full_layer_value_cache = getattr(self, f"full_value_cache_{idx}")
                new_quantize_layer_key_cache_ubits = getattr(self, f"quantize_key_cache_ubits{idx}")
                new_quantize_layer_key_cache_lbits = getattr(self, f"quantize_key_cache_lbits{idx}")
                new_trans_quantize_layer_value_cache_ubits = getattr(self, f"trans_quantize_value_cache_ubits{idx}")
                new_trans_quantize_layer_value_cache_lbits = getattr(self, f"trans_quantize_value_cache_lbits{idx}")
                new_min_layer_key_cache = getattr(self, f"min_key_cache_{idx}")
                new_trans_min_layer_value_cache = getattr(self, f"trans_min_value_cache_{idx}")
                new_scale_layer_key_cache = getattr(self, f"scale_key_cache_{idx}")
                new_trans_scale_layer_value_cache = getattr(self, f"trans_scale_value_cache_{idx}")

                torch._dynamo.mark_static_address(new_full_layer_key_cache)
                torch._dynamo.mark_static_address(new_full_layer_value_cache)
                torch._dynamo.mark_static_address(new_quantize_layer_key_cache_ubits)
                torch._dynamo.mark_static_address(new_quantize_layer_key_cache_lbits)
                torch._dynamo.mark_static_address(new_trans_quantize_layer_value_cache_ubits)
                torch._dynamo.mark_static_address(new_trans_quantize_layer_value_cache_lbits)

                torch._dynamo.mark_static_address(new_min_layer_key_cache)
                torch._dynamo.mark_static_address(new_trans_min_layer_value_cache)
                torch._dynamo.mark_static_address(new_scale_layer_key_cache)
                torch._dynamo.mark_static_address(new_trans_scale_layer_value_cache)

                self.full_key_cache.append(new_full_layer_key_cache)
                self.full_value_cache.append(new_full_layer_value_cache)
                self.quantize_key_cache_ubits.append(new_quantize_layer_key_cache_ubits)
                self.quantize_key_cache_lbits.append(new_quantize_layer_key_cache_lbits)
                self.trans_quantize_value_cache_ubits.append(new_trans_quantize_layer_value_cache_ubits)
                self.trans_quantize_value_cache_lbits.append(new_trans_quantize_layer_value_cache_lbits)

                self.min_key_cache.append(new_min_layer_key_cache)
                self.trans_min_value_cache.append(new_trans_min_layer_value_cache)
                self.scale_key_cache.append(new_scale_layer_key_cache)
                self.trans_scale_value_cache.append(new_trans_scale_layer_value_cache)

        self.register_buffer("seq_len", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("quant_key_len", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("quant_val_len", torch.tensor(0, dtype=torch.long, device=self.device))

        self.reset()

    def reset(self):
        self.seq_len.zero_()
        self.quant_key_len.zero_()
        self.quant_val_len.zero_()

        for idx in range(self.layers):
            self.full_key_cache[idx].zero_()
            self.full_value_cache[idx].zero_()
            self.quantize_key_cache_ubits[idx].zero_()
            self.quantize_key_cache_lbits[idx].zero_()
            self.trans_quantize_value_cache_ubits[idx].zero_()
            self.trans_quantize_value_cache_lbits[idx].zero_()

            self.min_key_cache[idx].zero_()
            self.trans_min_value_cache[idx].zero_()
            self.scale_key_cache[idx].zero_()
            self.trans_scale_value_cache[idx].zero_()
    
    def print_status(self):
        print("[Full Cache] Cached:", self.seq_len.item(), "| Quantized:", self.quant_key_len.item(), self.quant_val_len.item(), "| full_precision:", self.seq_len.item()-self.quant_key_len.item(), self.seq_len.item()-self.quant_val_len.item())

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        if layer_idx < self.first_few_fp16_layers:
            self.full_key_cache_fp16[layer_idx][:, :, self.seq_len:self.seq_len+key_states.shape[-2]].copy_(key_states)
            self.full_value_cache_fp16[layer_idx][:, :, self.seq_len:self.seq_len+value_states.shape[-2]].copy_(value_states)
            output = (
                self.full_key_cache_fp16[layer_idx][:, :, :self.seq_len+key_states.shape[-2]],
                None, None, None,
            )
            output += (
                self.full_value_cache_fp16[layer_idx][:, :, :self.seq_len+value_states.shape[-2]],
                None, None, None,
            )
            if layer_idx == self.layers:
                self.seq_len += key_states.shape[-2]
            return output

        elif self.seq_len.item() == 0: # prefill
            layer_idx -= self.first_few_fp16_layers
            output = (key_states, None, None, None)
            output += (value_states, None, None, None)

            prefill_len = key_states.shape[-2]
            assert prefill_len == value_states.shape[-2], "Key and value states must have the same length in prefill"
            if prefill_len > self.residual_length:
                assert self.residual_length % self.group_size == 0, "Residual length must be divisible by group size"
                to_quant_len = prefill_len-(prefill_len%self.residual_length)
                if prefill_len-to_quant_len > 0:

                    self.full_key_cache[layer_idx][:, :, :prefill_len-to_quant_len].copy_(key_states[:, :, to_quant_len:])
                    self.full_value_cache[layer_idx][:, :, :prefill_len-to_quant_len].copy_(value_states[:, :, to_quant_len:])

                key_states_quant_ubits, key_states_quant_lbits, key_scale, key_min = self.quantize_and_pack_along_penultimate_dim(key_states[:, :, :to_quant_len], self.group_size, self.k_bits)
                value_states_quant_ubits, value_states_quant_lbits, value_scale, value_min = self.quantize_and_pack_along_penultimate_dim(value_states[:, :, :to_quant_len], self.group_size, self.v_bits)
                self.quantize_key_cache_ubits[layer_idx][:, :, :to_quant_len].copy_(key_states_quant_ubits)
                self.quantize_key_cache_lbits[layer_idx][:, :, :to_quant_len].copy_(key_states_quant_lbits)
                self.trans_quantize_value_cache_ubits[layer_idx][:, :, :, :to_quant_len].copy_(value_states_quant_ubits)
                self.trans_quantize_value_cache_lbits[layer_idx][:, :, :, :to_quant_len].copy_(value_states_quant_lbits)

                self.scale_key_cache[layer_idx][:, :, :to_quant_len//self.group_size].copy_(key_scale)
                self.min_key_cache[layer_idx][:, :, :to_quant_len//self.group_size].copy_(key_min)
                self.trans_scale_value_cache[layer_idx][:, :, :, :to_quant_len].copy_(value_scale)
                self.trans_min_value_cache[layer_idx][:, :, :, :to_quant_len].copy_(value_min)

                if layer_idx == self.layers-self.first_few_fp16_layers-1:
                    self.quant_key_len += to_quant_len
                    self.quant_val_len += to_quant_len
                    self.seq_len += prefill_len
            return output
        else: # decode
            layer_idx -= self.first_few_fp16_layers
            self.full_key_cache[layer_idx][:, :, self.seq_len-self.quant_key_len:self.seq_len-self.quant_key_len+key_states.shape[-2]].copy_(key_states)
            self.full_value_cache[layer_idx][:, :, self.seq_len-self.quant_val_len:self.seq_len-self.quant_val_len+value_states.shape[-2]].copy_(value_states)
            output = (
                self.full_key_cache[layer_idx][:, :, :self.seq_len-self.quant_key_len+key_states.shape[-2]],
                self.quantize_key_cache_ubits[layer_idx][:, :, :self.quant_key_len],
                self.quantize_key_cache_lbits[layer_idx][:, :, :self.quant_key_len],
                self.scale_key_cache[layer_idx][:, :, :self.quant_key_len//self.group_size],
                self.min_key_cache[layer_idx][:, :, :self.quant_key_len//self.group_size],
            )
            output += (
                self.full_value_cache[layer_idx][:, :, :self.seq_len-self.quant_val_len+value_states.shape[-2]],
                self.trans_quantize_value_cache_ubits[layer_idx][:, :, :, :self.quant_val_len],
                self.trans_quantize_value_cache_lbits[layer_idx][:, :, :, :self.quant_val_len],
                self.trans_scale_value_cache[layer_idx][:, :, :, :self.quant_val_len],
                self.trans_min_value_cache[layer_idx][:, :, :, :self.quant_val_len],
            )
            if layer_idx == self.layers-self.first_few_fp16_layers-1:
                self.seq_len += key_states.shape[-2]
            return output

    def quantize_and_pack(self):
        full_key_len = self.seq_len - self.quant_key_len
        if full_key_len > self.residual_length:
            assert self.residual_length % self.group_size == 0, "Residual length must be divisible by group size"
            to_quant_len = full_key_len-(full_key_len%self.residual_length)
            for layer_idx in range(self.layers-self.first_few_fp16_layers):
                to_quant_key_states = self.full_key_cache[layer_idx][:, :, :to_quant_len]
                key_states_quant_ubits, key_states_quant_lbits, key_scale, key_min = self.quantize_and_pack_along_penultimate_dim(to_quant_key_states, self.group_size, self.k_bits)
                self.full_key_cache[layer_idx][:, :, :full_key_len-to_quant_len].copy_(self.full_key_cache[layer_idx][:, :, to_quant_len:full_key_len].clone())

                self.quantize_key_cache_ubits[layer_idx][:, :, self.quant_key_len:(self.quant_key_len+to_quant_len)].copy_(key_states_quant_ubits)
                self.quantize_key_cache_lbits[layer_idx][:, :, self.quant_key_len:(self.quant_key_len+to_quant_len)].copy_(key_states_quant_lbits)
                self.scale_key_cache[layer_idx][:, :, self.quant_key_len//self.group_size:(self.quant_key_len+to_quant_len)//self.group_size].copy_(key_scale)
                self.min_key_cache[layer_idx][:, :, self.quant_key_len//self.group_size:(self.quant_key_len+to_quant_len)//self.group_size].copy_(key_min)
            self.quant_key_len += to_quant_len

        full_val_len = self.seq_len - self.quant_val_len
        if full_val_len > self.residual_length:
            assert self.residual_length % self.group_size == 0, "Residual length must be divisible by group size"
            to_quant_len = full_val_len-self.residual_length
            for layer_idx in range(self.layers-self.first_few_fp16_layers):
                to_quant_value_states = self.full_value_cache[layer_idx][:, :, :to_quant_len]
                value_states_quant_ubits, value_states_quant_lbits, value_scale, value_min = self.quantize_and_pack_along_penultimate_dim(to_quant_value_states, self.group_size, self.v_bits)
                self.full_value_cache[layer_idx][:, :, :full_val_len-to_quant_len].copy_(self.full_value_cache[layer_idx][:, :, to_quant_len:full_val_len].clone())

                self.trans_quantize_value_cache_ubits[layer_idx][:, :, :, self.quant_val_len:(self.quant_val_len+to_quant_len)].copy_(value_states_quant_ubits)
                self.trans_quantize_value_cache_lbits[layer_idx][:, :, :, self.quant_val_len:(self.quant_val_len+to_quant_len)].copy_(value_states_quant_lbits)
                self.trans_scale_value_cache[layer_idx][:, :, :, self.quant_val_len:(self.quant_val_len+to_quant_len)].copy_(value_scale)
                self.trans_min_value_cache[layer_idx][:, :, :, self.quant_val_len:(self.quant_val_len+to_quant_len)].copy_(value_min)
            self.quant_val_len += to_quant_len
        
    def quantize_and_pack_along_penultimate_dim(self, inp, group_size, bits):
        num_groups = inp.size(-2) // group_size
        inp = inp.reshape(*inp.shape[:-2], num_groups, group_size, -1)
        maxval = torch.max(inp, dim=-2).values
        minval = torch.min(inp, dim=-2).values
        scale = (maxval - minval)/(2**(bits//2) - 1)
        ubits = inp - minval.unsqueeze(-2)
        ubits = ubits / scale.unsqueeze(-2)
        q_ubits = torch.clip(ubits, min=0, max=2**(bits//2)-1).round().to(torch.int32)

        q_error = ubits - q_ubits
        lbits = q_error*(2**(bits//2))
        q_lbits = torch.clip(lbits, min=-2**(bits//2 - 1), max=2**(bits//2 - 1)-1).round().to(torch.int32)

        return q_ubits.reshape(*inp.shape[:-2], -1, inp.shape[-1]), q_lbits.reshape(*inp.shape[:-2], -1, inp.shape[-1]), scale, minval