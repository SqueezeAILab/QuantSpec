# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer, LlamaConfig
from tqdm import tqdm
from data.dataset import get_dataset
from models.cache import QCache, SimQCacheV2, SimQCache
from models.modelling_llama_simquantkv import LlamaForCausalLM_SimQuant
from models.modelling_llama_quantkv import LlamaForCausalLM_Quant
from utils.misc import seed_everything
import torch.nn as nn
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    # Model configuration
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='model')
    parser.add_argument('--attn', type=str, default='flash_attention_2', help='attention implementation')
    parser.add_argument('--cache_dir', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--prefill', type=int, default=2048, help='prefill')

    # Quantization settings
    parser.add_argument('--quantized', action='store_true', help='whether to use quantized draft')
    parser.add_argument('--simquant', action='store_true', help='whether to use simquant')
    parser.add_argument('--k_bits', type=int, default=4, help='k_bits')
    parser.add_argument('--v_bits', type=int, default=4, help='v_bits')
    parser.add_argument('--group_size', type=int, default=32, help='group_size')
    parser.add_argument('--residual_length', type=int, default=32, help='residual_length')
    parser.add_argument('--num_examples', type=int, default=40, help='number of examples')
    parser.add_argument('--include_error', action='store_true', help='whether to include error in sensitivity')

    # Benchmarking and profiling
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--torch_profile', action='store_true', help='Use CUDA profiling tool for timing runs.')
    parser.add_argument('--filename', type=str, default='grads.json', help='filename')
    args = parser.parse_args()
    return args

def get_modules_kv(layer):
    k_proj, v_proj = layer.self_attn.k_proj, layer.self_attn.v_proj
    return k_proj, v_proj

def simulate_quant(data, bits=2, group_size=16):
    num_groups = data.size(-1) // group_size
    data = data.reshape(*data.shape[:-1], num_groups, group_size)
    maxval = torch.max(data, dim=-1).values
    minval = torch.min(data, dim=-1).values
    scale = (maxval - minval)/(2**bits - 1)
    q_data = data - minval.unsqueeze(-1)
    q_data = q_data / scale.unsqueeze(-1)
    q_data = torch.clip(q_data, min=0, max=2**bits-1).round().to(torch.int32)
    q_data_out = q_data*scale.unsqueeze(-1) + minval.unsqueeze(-1)
    return q_data_out.reshape(*data.shape[:-2], -1)


def llm_eval(model, cache, input_ids, args):
    print("Evaluating...")
    
    batch_size, total_length = input_ids.shape
    n_samples = total_length // args.prefill
    chunk_size = args.prefill  # Size of chunks for processing
    nlls = []
    grads = {}

    assert args.prefill % chunk_size == 0, "Prefill must be divisible by chunk size"    

    for sample_idx in tqdm(range(args.num_examples), desc="Evaluating samples"):
        # cache.reset()
        cache = SimQCache(model, max_budget=args.prefill+16)
        sample_start = sample_idx * args.prefill
        sample_end = (sample_idx + 1) * args.prefill
        input_ids_sample = input_ids[:, sample_start:sample_end].to(model.device)
        
        # with torch.no_grad():
        for chunk_start in range(0, args.prefill, chunk_size):
            for n, layer in enumerate(model.model.layers):
                k_proj, v_proj = get_modules_kv(layer)
                k_proj.retain_grad = True
                v_proj.retain_grad = True
            chunk_end = min(chunk_start + chunk_size, args.prefill)
            chunk_input = input_ids_sample[:, chunk_start:chunk_end]
            
            logits = model(input_ids=chunk_input, kv_cache=cache).logits
            
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_input[:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            model.zero_grad()
            loss.backward()
            # import IPython
            # IPython.embed()
            with torch.no_grad():
                for i, layer in enumerate(model.model.layers):
                    # print(f'weight layer {i}')
                    k_proj, v_proj = get_modules_kv(layer)
                    kgrad = (k_proj.act.grad ** 2).float().cpu().reshape(1, chunk_size, model.config.num_key_value_heads, model.config.head_dim) #.mean(-1).sum((0,1))
                    vgrad = (v_proj.act.grad ** 2).float().cpu().reshape(1, chunk_size, model.config.num_key_value_heads, model.config.head_dim) #.mean(-1).sum((0,1))


                    k = k_proj.act.data.reshape(1, chunk_size, model.config.num_key_value_heads, model.config.head_dim).float().cpu()
                    v = v_proj.act.data.reshape(1, chunk_size, model.config.num_key_value_heads, model.config.head_dim).float().cpu()
                    k_error = k - simulate_quant(k.clone().transpose(1, 3), bits=cache.k_bits, group_size=cache.group_size).transpose(1,3)
                    v_error = v - simulate_quant(v.clone(), bits=cache.v_bits, group_size=cache.group_size)
                    # import IPython
                    # IPython.embed()

                    assert k_error.shape == kgrad.shape
                    assert v_error.shape == vgrad.shape

                    if args.include_error:
                        sensitivity_k = (k_error**2)*kgrad
                        sensitivity_v = (v_error**2)*vgrad
                    else:
                        sensitivity_k = kgrad
                        sensitivity_v = vgrad

                    if f'k_proj{i}' not in grads:
                        grads[f'k_proj{i}'] = sensitivity_k.mean(-1).sum((0,1))
                    else:
                        grads[f'k_proj{i}'] += sensitivity_k.mean(-1).sum((0,1))
                    if f'v_proj{i}' not in grads:
                        grads[f'v_proj{i}'] = sensitivity_v.mean(-1).sum((0,1))
                    else:
                        grads[f'v_proj{i}'] += sensitivity_v.mean(-1).sum((0,1))    

            neg_log_likelihood = loss.float()
            nlls.append(neg_log_likelihood.detach().cpu())
            # import IPython
            # IPython.embed()
            
            # if args.quantized:
            #     cache.quantize_and_pack()
    for k, v in grads.items():
        grads[k] = v/(args.num_examples*args.prefill)

    # Save grads dictionary
    import json
    import os

    # Convert tensor values to float for JSON serialization
    grads_serializable = {k: v.numpy().tolist() for k, v in grads.items()}

    # Create a directory for saving results if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the grads dictionary to a JSON file
    with open(f'results/{args.filename}', 'w') as f:
        json.dump(grads_serializable, f, indent=4)

    print(f"Grads dictionary saved to results/{args.filename}")

    logppl = torch.stack(nlls).mean()
    
    print(f"PPL: {torch.exp(logppl).item():.4f}")
                


if __name__ == "__main__":
    seed_everything(43)
    args = parse_arguments()

    config = LlamaConfig.from_pretrained(args.model)
    config.use_cache = False
    config.k_bits = args.k_bits # KiVi currently support 2/4 K/V bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size 
    config._attn_implementation = args.attn
    config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
    CACHE_DIR = args.cache_dir
    config.precision='lower'
    # config.first_few_fp16_layers = 0
    config.first_few_fp16_layers = config.num_hidden_layers
    config.calculate_sensitivity = True
    # config.grads = None
    # config.k_threshold = None
    # config.v_threshold = None

    if args.simquant:
        model_class = LlamaForCausalLM_SimQuant
    else:
        model_class = LlamaForCausalLM_Quant  

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    
    
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        use_fast=True, 
        legacy=False,
        tokenizer_type='llama'
    )
    
    

    ####### cache init #######
    if args.simquant:
        cache = SimQCache(model, max_budget=args.prefill+16)
    else:
        cache = QCache(model, max_budget=args.prefill+16)

    for dataset in ["wikitext2"]:
        cache.reset()
        tokenized_prompts = get_dataset(dataset_name=dataset, tokenizer=tokenizer, datalen=args.prefill)
        llm_eval(model, cache, tokenized_prompts[0], args)
    
