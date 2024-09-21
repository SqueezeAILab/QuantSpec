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
from models.modeling_llama import LlamaForCausalLM_KIVI as LlamaForCausalLM_KIVI2
from models.modeling_llama_68m import LlamaForCausalLM as LlamaForCausalLM_68M
from models.cache import FlashSimpleCache, QCache, StreamingLLMEvictionCache, RetrievalCache
from utils.decoding import SpecDecode
from utils.misc import print_config
from utils.graph_infer import GraphInferenceEngine
from transformers import AutoTokenizer, LlamaConfig
from models.modelling_llama_kivi  import LlamaForCausalLM_KIVI

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')
    parser.add_argument('--cache_dir', type=str, default='/rscratch/rishabhtiwari/cache/')
    parser.add_argument('--quantized_draft', action='store_true', help='verbose')

    parser.add_argument('--target', type=str, default='llama-7B-128K', help='target model')
    parser.add_argument('--draft', type=str, default='llama-68M', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--prefill', type=int, default=32768, help='prefill length')
    parser.add_argument('--gen_len', type=int, default=256, help='generation length')
    parser.add_argument('--gamma', type=int, default=6, help='gamma')

    parser.add_argument('--dataset', type=str, default='gs', help='dataset')
    parser.add_argument('--temp', type=float, default=1., help='temperature')
    parser.add_argument('--top_p', type=float, default=-1., help='top p')
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--draft_cache_budget', type=int, default=256, help='draft cache budget')
    parser.add_argument('--chunk_size', type=int, default=8, help='chunk size')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_arguments()

    ######## model initialization ########
    # if args.target == 'llama-7B-128K':
    #     target = LlamaForCausalLM.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", cache_dir=args.cache_dir, torch_dtype=torch.float16, device_map="cuda:0")
    # else:
    #     raise NotImplementedError
    # target = target.eval()

    if args.quantized_draft:
        config = LlamaConfig.from_pretrained("JackFram/llama-68m")
        config.k_bits = 2 # KiVi currently support 2/4 K/V bits
        config.v_bits = 2
        config.group_size = 32 
        config.residual_length = 32 # corresponding to the number of recent fp16 tokens
        CACHE_DIR = args.cache_dir
        draft = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path='JackFram/llama-68m',
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
        )
        target = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path='JackFram/llama-68m',
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
        )
    else:
        draft = LlamaForCausalLM_68M.from_pretrained("JackFram/llama-68m", cache_dir=args.cache_dir, torch_dtype=torch.float16, device_map="cuda:0")
    
    draft = draft.eval()
    target = target.eval()

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-128k", cache_dir=args.cache_dir, use_fast=True, legacy=False)
    tokenizer = AutoTokenizer.from_pretrained(
    'JackFram/llama-68m', 
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
    chunk_size = args.chunk_size
    max_budget = args.budget

    # print_config(draft, None, prefill, gen_len, gamma, top_k, top_p, temperature, file_path=None, method="TriForce", spec_args={'budget': args.budget, 'chunk_size': chunk_size}, dataset=args.dataset)

    ####### cache init #######

    # draft_cache_budget = args.draft_cache_budget
    # recent_size = draft_cache_budget - 16 - gamma
    # cache = FlashSimpleCache(target, prefill+gen_len+16)
    # graph_cache = RetrievalCache(target, max_budget=max_budget, prefill=prefill, gamma=gamma, chunk_size=chunk_size)
    if args.quantized_draft:
        draft_cache = QCache(draft, max_budget=prefill+gen_len+16)
        target_cache = QCache(target, max_budget=prefill+gen_len+16)
    else:
        pass
        # draft_cache = StreamingLLMEvictionCache(draft, start_size=16, recent_size=recent_size, gamma=gamma)
    
    spec_engine = GraphInferenceEngine(target, target_cache, draft, draft_cache)

    # draft_cache.print_status()
    # target_cache.print_status()

    print(colored(f"tokenized_prompts length: {len(tokenized_prompts)}", "green"))


    # if args.quantized_draft:
    # ######## Testing quantization ########
    #     n_warmups = 1
    #     input_ids = tokenized_prompts[0].to(draft.device)[:,:prefill]
    #     for i in tqdm(range(n_warmups), desc="Speculative Decoding Warmup"):
    #         SpecDecode(tokenizer, spec_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)


    ######## Warm up for our method ########
    n_warmups = 3
    input_ids = tokenized_prompts[0].to(target.device)[:,:prefill]
    for i in tqdm(range(n_warmups), desc="TriForce Warmup"):
        SpecDecode(tokenizer, spec_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset)

    all_acceptance_rate = []
    all_speed = []
    for input_ids in tqdm(tokenized_prompts, desc="TriForce Test"):
        input_ids = input_ids.to(target.device)[:,:prefill]

        acceptance_rate, speed = SpecDecode(tokenizer, spec_engine, input_ids, gamma=gamma, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose, file_path=None, dataset=args.dataset)
        all_acceptance_rate.append(acceptance_rate)
        all_speed.append(speed)
    
    method_latency = 1000/(sum(all_speed) / len(all_speed))
    print(colored(f"average acceptance rate (NOT per token): {sum(all_acceptance_rate) / len(all_acceptance_rate)}", "red"))
    print(colored(f"[TriForce] average latency: {method_latency} ms", "red"))
