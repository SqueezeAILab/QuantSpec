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
from models.modelling_llama_simquantkivi import LlamaForCausalLM_SimQuant
from models.cache import SimUniformDynamicCache, SimNonUniformCache, QCache, SimQCache
from utils.decoding import SpecDecode, SimQuantSpecDecode
from utils.misc import print_config, log_csv, seed_everything
from utils.graph_infer import GraphInferenceEngine
from transformers import AutoTokenizer, LlamaConfig
from models.modelling_llama_kivi  import LlamaForCausalLM_KIVI445
import pickle

seed_everything(43)

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--cache_dir', type=str, default='/rscratch/rishabhtiwari/cache/')
    parser.add_argument('--quantized_draft', action='store_true', help='whether to use quantized draft')
    parser.add_argument('--simquant', action='store_true', help='whether to use simulated quantization')
    parser.add_argument('--k_bits', type=int, default=4, help='k_bits')
    parser.add_argument('--v_bits', type=int, default=4, help='v_bits')
    parser.add_argument('--group_size', type=int, default=32, help='group_size')
    parser.add_argument('--residual_length', type=int, default=32, help='residual_length')
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
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    seed_everything(43)
    args = parse_arguments()

    config = LlamaConfig.from_pretrained(args.draft)
    config.use_cache = False
    config.k_bits = args.k_bits # currently only support 4-bit
    config.v_bits = args.v_bits # currently only support 4-bit
    config.group_size = args.group_size 
    config.use_cache = False
    config._attn_implementation = args.attn
    config.residual_length = args.residual_length # corresponding to the number of recent fp16 tokens
    CACHE_DIR = args.cache_dir
    if args.simquant:
        draft_class = LlamaForCausalLM_SimQuant
        target_class = LlamaForCausalLM_SimQuant
    else:
        draft_class = LlamaForCausalLM_KIVI445
        target_class = LlamaForCausalLM_KIVI445
    draft = draft_class.from_pretrained(
    pretrained_model_name_or_path=args.draft,
    config=config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    device_map="cuda:0"
    )
    
    target = target_class.from_pretrained(
    pretrained_model_name_or_path=args.target,
    config=config,
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
        draft_cache = SimQCache(draft, max_budget=prefill+gen_len+16)
        target_cache = SimQCache(target, max_budget=prefill+gen_len+16)
    else:
        target_cache = QCache(target, max_budget=prefill+gen_len+16)
        draft_cache = target_cache
    
    spec_engine = GraphInferenceEngine(target, target_cache, draft, draft_cache)

    draft_cache.print_status()
    target_cache.print_status()

    print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))


    decoding_fn = SimQuantSpecDecode if args.quantized_draft else SpecDecode

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