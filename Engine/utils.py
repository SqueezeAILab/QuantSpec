import torch
import numpy as np
import random
from torch.nn.functional import softmax
from flash_attn import flash_attn_with_kvcache
from QuantSpec.kernels.flashdecoding.int8_verify_upperlower.int8kv_verify_upperlower_flash_decoding import token_decode_attention_int8kv_verify_upperlower_flash_decoding

from QuantSpec.kernels.flashdecoding.int8_upperlower.int8kv_upperlower_flash_decoding import token_decode_attention_int8kv_upperlower_flash_decoding

from marlin import marlin_cuda
import torch.distributed as dist
from gptqmodel.nn_modules.qlinear.marlin import apply_gptq_marlin_linear


torch.library.define(
    "mylib::gptq_marlin_linear",
    "(Tensor(a!) input, \
    Tensor(b!) weight, \
    Tensor(c!) weight_scale, \
    Tensor(d!) weight_zp, \
    Tensor(e!) g_idx, \
    Tensor(f!) g_idx_sort_indices, \
    Tensor(g!) workspace, \
    Scalar num_bits, \
    Scalar output_size_per_partition, \
    Scalar input_size_per_partition, \
    Scalar is_k_full, \
    Scalar fp32 \
    ) -> Tensor",
)

@torch.library.impl("mylib::gptq_marlin_linear", "cuda")
def gptq_marlin_linear(input, weight, weight_scale, weight_zp, g_idx, g_idx_sort_indices, workspace, num_bits, output_size_per_partition, input_size_per_partition, is_k_full, fp32):
    return apply_gptq_marlin_linear(input, weight, weight_scale, weight_zp, g_idx, g_idx_sort_indices, workspace, num_bits, output_size_per_partition, input_size_per_partition, is_k_full, None, True)


@torch.library.impl_abstract("mylib::gptq_marlin_linear")
def gptq_marlin_linear_abstract(input, weight, weight_scale, weight_zp, g_idx, g_idx_sort_indices, workspace, num_bits, output_size_per_partition, input_size_per_partition, is_k_full, fp32):
    output_shape = input.shape[:-1] + (output_size_per_partition,)
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)

torch.library.define(
    "mylib::marlin_mul",
    "(Tensor(a!) A, \
    Tensor(b!) B, \
    Tensor(c!) C, \
    Tensor(d!) s, \
    Tensor(e!) workspace \
    ) -> Tensor",
)

@torch.library.impl("mylib::marlin_mul", "cuda")
def marlin_mul(A, B, C, s, workspace):
    marlin_cuda.mul(A, B, C, s, workspace, -1, -1, -1, 16)


@torch.library.impl_abstract("mylib::marlin_mul")
def marlin_mul_abstract(A, B, C, s, workspace):
    return torch.empty(C.shape, dtype=C.dtype, device=C.device)

torch.library.define(
    "mylib::flash_verification",
    "(Tensor q, \
    Tensor(a!) cache_quant_k_upper, \
    Tensor(b!) cache_quant_k_lower, \
    Tensor(c!) cache_scale_k, \
    Tensor(d!) cache_min_k, \
    Tensor(e!) cache_quant_v_upper, \
    Tensor(f!) cache_quant_v_lower, \
    Tensor(g!) cache_scale_v, \
    Tensor(h!) cache_min_v, \
    Scalar kbit, \
    Scalar vbit, \
    Scalar group_size, \
    Tensor(i!) full_k, \
    Tensor(j!) full_v, \
    Scalar precision, \
    Scalar max_seq_length, \
    Scalar max_residual_len, \
    Tensor qcache_len, \
    Tensor residual_len) -> Tensor",
)

@torch.library.impl("mylib::flash_verification", "cuda")
def flash_verification(
    q,
    cache_quant_k_upper,
    cache_quant_k_lower,
    cache_scale_k,
    cache_min_k,
    cache_quant_v_upper,
    cache_quant_v_lower,
    cache_scale_v,
    cache_min_v,
    kbit,
    vbit,
    group_size,
    full_k,
    full_v,
    precision,
    max_seq_length,
    max_residual_len,
    qcache_len,
    residual_len
):
    return token_decode_attention_int8kv_verify_upperlower_flash_decoding(
        q=q,
        cache_quant_k_upper=cache_quant_k_upper,
        cache_quant_k_lower=cache_quant_k_lower,
        cache_scale_k=cache_scale_k,
        cache_min_k=cache_min_k,
        cache_quant_v_upper=cache_quant_v_upper,
        cache_quant_v_lower=cache_quant_v_lower,
        cache_scale_v=cache_scale_v,
        cache_min_v=cache_min_v,
        kbit=kbit,
        vbit=vbit,
        group_size=group_size,
        full_k=full_k,
        full_v=full_v,
        precision=precision,
        max_seq_length=max_seq_length,
        max_residual_len=max_residual_len,
        qcache_len=qcache_len,
        residual_len=residual_len
    )

@torch.library.impl_abstract("mylib::flash_verification")
def flash_verification_abstract(
    q,
    cache_quant_k_upper,
    cache_quant_k_lower,
    cache_scale_k,
    cache_min_k,
    cache_quant_v_upper,
    cache_quant_v_lower,
    cache_scale_v,
    cache_min_v,
    kbit,
    vbit,
    group_size,
    full_k,
    full_v,
    precision,
    max_seq_length,
    max_residual_len,
    qcache_len,
    residual_len
):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)

torch.library.define(
    "mylib::flash_decoding",
    "(Tensor q, \
    Tensor(a!) cache_quant_k_upper, \
    Tensor(b!) cache_quant_k_lower, \
    Tensor(c!) cache_scale_k, \
    Tensor(d!) cache_min_k, \
    Tensor(e!) cache_quant_v_upper, \
    Tensor(f!) cache_quant_v_lower, \
    Tensor(g!) cache_scale_v, \
    Tensor(h!) cache_min_v, \
    Scalar kbit, \
    Scalar vbit, \
    Scalar group_size, \
    Tensor(i!) full_k, \
    Tensor(j!) full_v, \
    Scalar precision, \
    Scalar max_seq_length, \
    Scalar max_residual_len, \
    Tensor qcache_len, \
    Tensor residual_len) -> Tensor",
)


@torch.library.impl("mylib::flash_decoding", "cuda")
def flash_decoding(
    q,
    cache_quant_k_upper,
    cache_quant_k_lower,
    cache_scale_k,
    cache_min_k,
    cache_quant_v_upper,
    cache_quant_v_lower,
    cache_scale_v,
    cache_min_v,
    kbit,
    vbit,
    group_size,
    full_k,
    full_v,
    precision,
    max_seq_length,
    max_residual_len,
    qcache_len,
    residual_len
):
    return token_decode_attention_int8kv_upperlower_flash_decoding(
        q=q,
        cache_quant_k_upper=cache_quant_k_upper,
        cache_quant_k_lower=cache_quant_k_lower,
        cache_scale_k=cache_scale_k,
        cache_min_k=cache_min_k,
        cache_quant_v_upper=cache_quant_v_upper,
        cache_quant_v_lower=cache_quant_v_lower,
        cache_scale_v=cache_scale_v,
        cache_min_v=cache_min_v,
        kbit=kbit,
        vbit=vbit,
        group_size=group_size,
        full_k=full_k,
        full_v=full_v,
        precision=precision,
        max_seq_length=max_seq_length,
        max_residual_len=max_residual_len,
        qcache_len=qcache_len,
        residual_len=residual_len
    )

@torch.library.impl_abstract("mylib::flash_decoding")
def flash_decoding_abstract(
    q,
    cache_quant_k_upper,
    cache_quant_k_lower,
    cache_scale_k,
    cache_min_k,
    cache_quant_v_upper,
    cache_quant_v_lower,
    cache_scale_v,
    cache_min_v,
    kbit,
    vbit,
    group_size,
    full_k,
    full_v,
    precision,
    max_seq_length,
    max_residual_len,
    qcache_len,
    residual_len
):
    return torch.empty_like(q)



torch.library.define(
    "mylib::custom_func",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl("mylib::custom_func", "cuda")
def custom_func(q, k_cache, v_cache, k, v, cache_seqlens):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True
    )

@torch.library.impl_abstract("mylib::custom_func")
def custom_func_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    return torch.empty_like(q)

torch.library.define(
    "mylib::custom_func_2",
    "(Tensor q, Tensor(a!) k_cache, Tensor(a!) v_cache) -> Tensor",
)

@torch.library.impl("mylib::custom_func_2", "cuda")
def custom_func_2(q, k_cache, v_cache):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, causal=True
    )

@torch.library.impl_abstract("mylib::custom_func_2")
def custom_func_2_abstract(q, k_cache, v_cache):
    return torch.empty_like(q)

torch.library.define(
    "mylib::gqa_custom",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl_abstract("mylib::gqa_custom")
def gqa_custom_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    return torch.empty_like(q)

@torch.library.impl("mylib::gqa_custom", "cuda")
def gqa_custom(q, k_cache, v_cache, k, v, cache_seqlens):
    B, T, H_q, D = q.size()
    H_k = k.size(2)
    rep = H_q // H_k
    q_reshaped = q.view(B, T, H_k, rep, D).transpose(2, 3).contiguous().view(B, T*rep, H_k, D).contiguous()
    v_new = torch.zeros(B, T*rep, H_k, D, device=q.device, dtype=q.dtype)
    k_new = torch.zeros_like(v_new)
    
    # the extra 1's added to the partition functions
    # they are of the pattern [0, 1, 2, ..., rep-1, rep-1, rep, rep+1, ..., 2*rep-1, 2*rep-1, 2*rep, ...]
    offset = torch.ones(rep, device=q.device, dtype=q.dtype)
    offset[0].zero_()
    extra = torch.cumsum(offset.repeat(T), dim=0)[None, None, :]
    insert_indices = torch.arange(0, T*rep, rep, device=q.device)[None, :, None, None].expand(B, -1, H_k, D)
    k_new.scatter_(1, insert_indices, k)
    v_new.scatter_(1, insert_indices, v)
    
    # print(q_reshaped.shape, k_cache.shape, k_new.shape)
    y, lse = flash_attn_with_kvcache(q_reshaped, k_cache, v_cache, k_new, v_new, cache_seqlens=cache_seqlens, causal=True, return_softmax_lse=True)
    
    extra = extra.expand_as(lse)
    correction = 1./ (1 - extra * torch.exp(-lse))
    correction = correction.transpose(1, 2).unsqueeze(-1)
    y = y * correction.to(y.dtype)
    y = y.view(B, T, rep, H_k, D).transpose(2, 3).contiguous().view(B, T, H_q, D)
    
    # insert new k and v to k_cache and v_cache, starting from cache_seqlens position
    insert_indices = cache_seqlens.unsqueeze(-1) + torch.arange(T, device=cache_seqlens.device).unsqueeze(0)
    insert_indices = insert_indices[..., None, None].expand(-1, -1, H_k, D)
    k_cache.scatter_(1, insert_indices, k)
    v_cache.scatter_(1, insert_indices, v)   

    return y.to(q.dtype)

def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    shape = logits.shape
    if top_p < 1.0:
        if len(shape)==3:
            batch_size, seq_len, voc_size = logits.size()
            logits = logits.reshape(-1, voc_size)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(-1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
        if len(shape)==3:
            logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def sample(logits, top_p, T):
    shape = logits.shape
    if len(shape)==3:
        batch_size, seq_len, _ = logits.size()
    else:
        batch_size, _ = logits.size()
        seq_len = 1
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, logits.shape[-1]).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cg_get_sampling_logits(logits :torch.Tensor, top_p:float, T: float):
    logits = logits.clone()
    batch_size, seq_len, voc_size = logits.size()
    logits = logits.reshape(-1, voc_size)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
    filter = cumulative_probs > top_p
    filter[..., 1:] = filter[..., :-1].clone()
    filter[..., 0] = 0
    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
    logits[indices_to_remove] = float('-inf')
    logits = logits.reshape(batch_size, seq_len, voc_size)
    return logits

def cg_sample(logits, top_p, T):
    batch_size, seq_len, _ = logits.size()
    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T, replicate=True)
    logits = softmax(logits / T, dim=-1)
    next_tokens = logits.view(-1, logits.shape[-1]).multinomial(num_samples=1).view(batch_size, seq_len)
    return next_tokens

def cuda_graph_for_target_sample(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 3, batch_size=1, top_p = 0.9, T = 0.6):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_tokens = cg_sample(
                 static_sampling_logits,
                 top_p=top_p, T=T
            )
    def run(target_logits, top_p=None, T=None):
        static_sampling_logits.copy_(target_logits)
        graph.replay()
        return static_tokens.clone()
    return run

def sampling_argmax_batch(logits: torch.Tensor):
    return logits.topk(k=1, dim=-1).indices.flatten(start_dim=1).long()

def cuda_graph_for_sampling_argmax_batch(
                device="cuda:0", dtype=torch.bfloat16, 
                dim=32000, n_warmups=3, mempool=None,
                idx_len = 1, batch_size=1):
    
    static_sampling_logits = torch.full((batch_size, idx_len, dim), 1, dtype=dtype, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax_batch(
                 static_sampling_logits,
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    return run

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model(checkpoint_path, device, precision, use_tp, rank_group=None, group=None):
    from QuantSpec.Engine.model import Transformer
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from QuantSpec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def add_marlin_dict(checkpoint, marlin_dict, replacements=None):
    if replacements is None:
        replacements = {
            'mlp': 'feed_forward',
            'model.layers': 'layers',
            'gate_proj': 'w1_quantized',
            'up_proj': 'w3_quantized',
            'down_proj': 'w2_quantized'
        }

    for key in marlin_dict:
        if 'mlp' in key:
            engine_key = key
            for old, new in replacements.items():
                if old in engine_key:
                    engine_key = engine_key.replace(old, new)
            checkpoint[engine_key] = marlin_dict[key]

    return checkpoint

def load_model_quantspec(checkpoint_path, device, precision, use_tp, rank_group=None, group=None, quantize: bool = False, marlin_checkpoint: str = None):
    import QuantSpec.Engine.model_quantspec as quantspec
    with torch.device(device):
        model = quantspec.Transformer.from_name(checkpoint_path.parent.name, quantize=quantize)
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    if quantize:
        marlin_dict = torch.load(marlin_checkpoint, mmap=True, weights_only=True)
        checkpoint = add_marlin_dict(checkpoint, marlin_dict)

    model.load_state_dict(checkpoint, assign=True)


    if use_tp:
        from QuantSpec.Engine.tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model, rank_group, group=group)
    
    model = model.to(device=device, dtype=precision)
    return model.eval()