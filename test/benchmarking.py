import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from transformers import AutoTokenizer, LlamaConfig
from data.dataset import get_dataset
import torch
from models.modelling_llama_simquant import LlamaForCausalLM_Simquant
from utils.decoding import Autoregressive
from models.cache import SimUniformDynamicCache
from utils.graph_infer import GraphInferenceEngine
from tqdm import tqdm
import pdb




config = LlamaConfig.from_pretrained('/rscratch/llama-7b/')
config.k_bits = 4 # KiVi currently support 2/4 K/V bits
config.v_bits = 4
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
CACHE_DIR = '/rscratch/rishabhtiwari/cache/'
pdb.set_trace()

draft = LlamaForCausalLM_Simquant.from_pretrained(
pretrained_model_name_or_path='/rscratch/llama-7b/',
config=config,
cache_dir=CACHE_DIR,
torch_dtype=torch.float16,
device_map="cuda:0"
)
draft = draft.eval()

tokenizer = AutoTokenizer.from_pretrained(
    '/rscratch/llama-7b/', 
    use_fast=True, 
    legacy=False,
    tokenizer_type='llama')
prefill = 16000
gen_len = 256
top_k = -1
top_p = 0.9
temperature = 0.6
verbose = False

tokenized_prompts = get_dataset(dataset_name='gs', tokenizer=tokenizer, datalen=prefill)

draft_cache = SimUniformDynamicCache(draft, max_budget=prefill+gen_len+16)

draft_cache.print_status()

spec_engine = GraphInferenceEngine(draft, draft_cache, draft, draft_cache)

n_warmups = 3
input_ids = tokenized_prompts[0].to(draft.device)[:,:prefill]
for i in tqdm(range(n_warmups), desc="Warmup"):
    Autoregressive(tokenizer, spec_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)

breakpoint()
all_speed = []
for input_ids in tqdm(tokenized_prompts, desc="Speed Test"):
    input_ids = input_ids.to(draft.device)[:,:prefill]

    speed = Autoregressive(tokenizer, spec_engine, input_ids, max_len=gen_len, top_k=top_k, top_p=top_p, temperature=temperature, verbose=verbose)
    breakpoint()
    all_speed.append(speed)

print(f"Average Speed: {sum(all_speed) / len(all_speed)}")

