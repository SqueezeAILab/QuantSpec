# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer, LlamaConfig
from tqdm import tqdm
from data.dataset import get_dataset
from models.modelling_llama_simquantkivi import LlamaForCausalLM_KIVI445 as LlamaForCausalLM_SimKIVI445
from models.cache import QCache, SimQCache
from models.modelling_llama_kivi import LlamaForCausalLM_KIVI445
from utils.misc import seed_everything
import torch.nn as nn
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    # Model configuration
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='model')
    parser.add_argument('--attn', type=str, default='flash_attention_2', help='attention implementation')
    parser.add_argument('--cache_dir', type=str, default='/rscratch/rishabhtiwari/cache/')
    parser.add_argument('--prefill', type=int, default=2048, help='prefill')

    # Quantization settings
    parser.add_argument('--quantized', action='store_true', help='whether to use quantized draft')
    parser.add_argument('--simquant', action='store_true', help='whether to use simquant')
    parser.add_argument('--k_bits', type=int, default=4, help='k_bits')
    parser.add_argument('--v_bits', type=int, default=4, help='v_bits')
    parser.add_argument('--group_size', type=int, default=32, help='group_size')
    parser.add_argument('--residual_length', type=int, default=32, help='residual_length')

    # Benchmarking and profiling
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--torch_profile', action='store_true', help='Use CUDA profiling tool for timing runs.')
    args = parser.parse_args()
    return args

def llm_eval(model, cache, input_ids, args):
    print("Evaluating...")
    
    batch_size, total_length = input_ids.shape
    n_samples = total_length // args.prefill
    chunk_size = 128  # Size of chunks for processing
    nlls = []

    assert args.prefill % chunk_size == 0, "Prefill must be divisible by chunk size"    

    for sample_idx in tqdm(range(n_samples), desc="Evaluating samples"):
        cache.reset()
        sample_start = sample_idx * args.prefill
        sample_end = (sample_idx + 1) * args.prefill
        input_ids_sample = input_ids[:, sample_start:sample_end].to(model.device)
        
        with torch.no_grad():
            for chunk_start in range(0, args.prefill, chunk_size):
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
                
                neg_log_likelihood = loss.float()
                nlls.append(neg_log_likelihood)
                # import IPython
                # IPython.embed()
                
                if args.quantized:
                    cache.quantize_and_pack()
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

    if args.quantized:
        model_class = LlamaForCausalLM_SimKIVI445
    else:
        model_class = LlamaForCausalLM_KIVI445  

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
        tokenized_prompts = get_dataset(dataset_name=dataset, tokenizer=tokenizer, datalen=args.prefill)
        llm_eval(model, cache, tokenized_prompts[0], args)
    
