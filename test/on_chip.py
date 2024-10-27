# CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from transformers import AutoTokenizer
from termcolor import colored
from tqdm import tqdm
from data.dataset import get_dataset
from models.modelling_llama_simquantkv import LlamaForCausalLM_SimQuant
from models.modelling_llama_quantkv import LlamaForCausalLM_Quant
from models.cache import QCache, SimQCacheV2
from utils.decoding import SimQuantSpecDecode
from utils.misc import print_config, log_csv, seed_everything
from utils.graph_infer import GraphInferenceEngine
from transformers import AutoTokenizer, LlamaConfig
import numpy as np
import copy
seed_everything(43)

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--cache_dir', type=str, default='/rscratch/rishabhtiwari/cache/')
    parser.add_argument('--quantized_draft', action='store_true', help='whether to use quantized draft')
    parser.add_argument('--simquant', action='store_true', help='whether to use simulated quantization')
    parser.add_argument('--first_few_fp16_layers', type=int, default=0, help='first few fp16 layers')
    parser.add_argument('--k_bits', type=int, default=4, help='k_bits')
    parser.add_argument('--v_bits', type=int, default=4, help='v_bits')
    parser.add_argument('--group_size', type=int, default=128, help='group_size')
    parser.add_argument('--residual_length', type=int, default=128, help='residual_length')
    parser.add_argument('--attn', type=str, default='flash_attention_2', help='attention implementation')

    parser.add_argument('--target', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='target model')
    parser.add_argument('--draft', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top p')
    parser.add_argument('--log_file', type=str, default='on_chip_results.txt', help='log file')
    parser.add_argument('--identifier_string', type=str, default='', help='identifier string')

    parser.add_argument('--mixed_precision', action='store_true', help='mixed precision')
    parser.add_argument('--sensitivity_filepath', type=str, default='', help='sensitivity filepath')
    parser.add_argument('--percentile', type=float, default=0.05, help='fraction of attention heads/layers in higher precision')
    parser.add_argument('--kv_combined_threshold', action='store_true', help='combined threshold for k and v')
    parser.add_argument('--per_head_threshold', action='store_true', help='per head threshold')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    seed_everything(43)
    args = parse_arguments()

    config = LlamaConfig.from_pretrained(args.draft)
    config.use_cache = False
    config.calculate_sensitivity = True
    config.k_bits = args.k_bits # currently only support 4-bit
    config.v_bits = args.v_bits # currently only support 4-bit
    config.first_few_fp16_layers = args.first_few_fp16_layers # number of initial layers to keep in fp16
    config.group_size = args.group_size 
    config.use_cache = False
    config._attn_implementation = args.attn
    config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
    
    config.mixed_precision = None if not args.mixed_precision else args.mixed_precision
    config.precision='higher'

    target_config = copy.deepcopy(config)

    if args.mixed_precision:
        import json
        import os
        sensitivity_filepath = args.sensitivity_filepath
        if os.path.exists(sensitivity_filepath):
            with open(sensitivity_filepath, 'r') as f:
                sensitivity_data = json.load(f)
            if args.per_head_threshold:
                k_sensitivity = [sensitivity_data[f'k_proj{i}'][j] for i in range(config.first_few_fp16_layers, config.num_hidden_layers) for j in range(config.num_key_value_heads)]
                v_sensitivity = [sensitivity_data[f'v_proj{i}'][j] for i in range(config.first_few_fp16_layers, config.num_hidden_layers) for j in range(config.num_key_value_heads)]
            else:
                k_sensitivity = [np.mean(sensitivity_data[f'k_proj{i}']) for i in range(config.first_few_fp16_layers, config.num_hidden_layers)]
                v_sensitivity = [np.mean(sensitivity_data[f'v_proj{i}']) for i in range(config.first_few_fp16_layers, config.num_hidden_layers)]
            
            if args.kv_combined_threshold:
                kv_sensitivity = k_sensitivity + v_sensitivity
                sorted_kv_sensitivity = sorted(kv_sensitivity, reverse=True)
                threshold_index = int(args.percentile * len(sorted_kv_sensitivity))
                threshold = sorted_kv_sensitivity[threshold_index]
                k_threshold = threshold
                v_threshold = threshold
            else:
                k_threshold = np.percentile(k_sensitivity, args.percentile)
                v_threshold = np.percentile(v_sensitivity, args.percentile)
            
            mixed_precision_kbits = []
            mixed_precision_vbits = []
            for i in range(config.first_few_fp16_layers, config.num_hidden_layers):
                if args.per_head_threshold:
                    mixed_precision_kbits.append([args.k_bits if sensitivity_data[f'k_proj{i}'][j] > k_threshold else 2*args.k_bits for j in range(config.num_key_value_heads)])
                    mixed_precision_vbits.append([args.v_bits if sensitivity_data[f'v_proj{i}'][j] > v_threshold else 2*args.v_bits for j in range(config.num_key_value_heads)])
                else:
                    mixed_precision_kbits.append([args.k_bits if np.mean(sensitivity_data[f'k_proj{i}']) > k_threshold else 2*args.k_bits]*config.num_key_value_heads)
                    mixed_precision_vbits.append([args.v_bits if np.mean(sensitivity_data[f'v_proj{i}']) > v_threshold else 2*args.v_bits]*config.num_key_value_heads)     
            config.k_mixed_bits = mixed_precision_kbits
            config.v_mixed_bits = mixed_precision_vbits
            
        else:
            raise ValueError(f"Using mixed precision but sensitivity file {sensitivity_filepath} does not exist")

    CACHE_DIR = args.cache_dir

    if args.simquant:
        draft_class = LlamaForCausalLM_SimQuant
        target_class = LlamaForCausalLM_SimQuant
    else:
        draft_class = LlamaForCausalLM_Quant
        target_class = LlamaForCausalLM_Quant
    draft = draft_class.from_pretrained(
    pretrained_model_name_or_path=args.draft,
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="cuda:0"
    )
    
    target_config = copy.deepcopy(config)
    target_config.first_few_fp16_layers = config.num_hidden_layers
    target = target_class.from_pretrained(
    pretrained_model_name_or_path=args.target,
    config=target_config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="cuda:0"
    )
    
    draft = draft.eval()
    target = target.eval()

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", cache_dir=args.cache_dir, use_fast=True, legacy=False)
    tokenizer = AutoTokenizer.from_pretrained(
    args.target, 
    use_fast=True, 
    legacy=False,
    tokenizer_type='llama')
    tokenized_prompts = get_dataset(dataset_name=args.dataset, tokenizer=tokenizer, datalen=args.prefill)

    ######## sampling parameters ########

    top_k = -1
    top_p = args.top_p
    temperature = args.temp

    prefill = args.prefill
    gen_len = args.gen_len
    gamma = args.gamma
    verbose = args.verbose
    log_file = args.log_file
    print_config(draft, target, args.dataset, prefill, gen_len, gamma, top_k, top_p, temperature, args.quantized_draft, args.k_bits, args.v_bits, args.group_size, args.residual_length)

    ####### cache init #######
    if args.simquant:
        draft_cache = SimQCacheV2(draft, max_budget=prefill+gen_len+16)
        target_cache = SimQCacheV2(target, max_budget=prefill+gen_len+16)
    else:
        target_cache = QCache(target, max_budget=prefill+gen_len+16)
        draft_cache = QCache(draft, max_budget=prefill+gen_len+16)
    
    spec_engine = GraphInferenceEngine(target, target_cache, draft, draft_cache)

    draft_cache.print_status()
    target_cache.print_status()

    print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))


    decoding_fn = SimQuantSpecDecode

    log_csv(log_file, "average_acceptance_rate, acceptance_rate_list, average_latency\n", f"{args.identifier_string}, {args.dataset}, {args.prefill}, {args.k_bits}, {args.v_bits}, {args.group_size}\n")




    ######## Warm up for our method ########
    # n_warmups = 1
    # input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
    # for i in tqdm(range(n_warmups), desc="SpecDecode Warmup"):
    #     decoding_fn(tokenizer, spec_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset)

    all_acceptance_rate = []
    all_speed = []
    for input_ids in tqdm(tokenized_prompts[:5], desc="SpecDecode Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]

        acceptance_rate, speed = decoding_fn(tokenizer, spec_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset)
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)
    
    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[SpecQuant] average latency: {method_latency} ms", "red"))
    
    log_csv(log_file, None, f"{sum(all_acceptance_rate) / len(all_acceptance_rate)}, {str(all_acceptance_rate)}, {method_latency}\n")