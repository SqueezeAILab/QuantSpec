import torch
import math
import time
import numpy as np
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.misc import spec_stream, log_csv
from utils.sampling import sample, norm_logits, max_fn
        

@torch.inference_mode()
def SimQuantSpecDecode(tokenizer, spec_engine, input_ids, gamma, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None, spec_args=None):
    spec_engine.clear_kv()

    logits = spec_engine.inference(input_ids=input_ids)
    _ = spec_engine.draft_prefill(input_ids=input_ids)
    spec_engine.engine.draft_cache.quantize_and_pack()

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    accepted_count = 0
    draft_count = 0
    resample_count = 0
    n = 0
    time1 = time.time()
    while n < max_len:
        pred_token_idx = next_token.squeeze()
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        verify_tokens = torch.full((1, gamma+1), 100, device=spec_engine.engine.draft.device)
        verify_tokens[:, 0] = next_token
        speculation_probs = []
        for i in range(gamma):
            # print(i)
            spec_engine.engine.draft_cache.seq_len -= i
            logits_ = spec_engine.draft_inference(input_ids=verify_tokens[:,:i+1])
            # spec_engine.engine.draft_cache.print_status()
            # logits_ = spec_engine.draft_inference(input_ids=verify_tokens[:,i:i+1])
            prob = norm_logits(logits_[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p).squeeze()
            speculation_probs.append(prob)
            next_token = sample(prob)
            verify_tokens[:, i+1] = next_token.item()
            # draft_count += 1
        draft_count += gamma
        
        logits = spec_engine.inference(input_ids=verify_tokens)
        count = 0  
        verify_probs = []
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma + 1):
            verify_probs.append(probs[i])
        
        for i, speculation_prob, verify_prob in zip(verify_tokens[0,1:], speculation_probs, verify_probs):
            r = torch.rand(1, device = spec_engine.engine.target.device)
            # print(r, verify_prob[i] / speculation_prob[i])
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                if verbose:
                    spec_stream(i, tokenizer, 'green')
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    print("EOS"*20, flush=True)
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

            # if tokenizer.eos_token_id == pred_token_idx:
            #     break
        
        spec_engine.engine.target_cache.seq_len -= (gamma - count)
        spec_engine.engine.draft_cache.seq_len -= max(0,(gamma - count - 1))

        if count == gamma:
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            # spec_engine.engine.draft_cache.seq_len -= gamma
            # spec_engine.draft_inference(input_ids=verify_tokens)
            spec_engine.draft_inference(input_ids=verify_tokens[:,-1:])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            count += 1

        spec_engine.engine.draft_cache.quantize_and_pack()

        next_token = pred_token_idx
    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {spec_engine.engine.target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate, n / (time2 - time1)

@torch.inference_mode()
def QuantSpecDecode(tokenizer, spec_engine, input_ids, gamma, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None, spec_args=None):
    spec_engine.clear_kv()

    logits = spec_engine.inference(input_ids=input_ids)
    # spec_engine.engine.target_cache.quantize_and_pack()

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    accepted_count = 0
    draft_count = 0
    resample_count = 0
    n = 0
    time1 = time.time()
    while n < max_len:
        pred_token_idx = next_token.squeeze()
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        verify_tokens = torch.full((1, gamma+1), 100, device=spec_engine.engine.draft.device)
        verify_tokens[:, 0] = next_token
        speculation_probs = []
        for i in range(gamma):
            # print(i)
            # spec_engine.engine.draft_cache.seq_len -= i
            # logits_ = spec_engine.draft_inference(input_ids=verify_tokens[:,:i+1])
            # spec_engine.engine.draft_cache.print_status()
            logits_ = spec_engine.draft_inference(input_ids=verify_tokens[:,i:i+1])
            prob = norm_logits(logits_[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p).squeeze()
            speculation_probs.append(prob)
            next_token = sample(prob)
            verify_tokens[:, i+1] = next_token.item()
            # draft_count += 1
        draft_count += gamma


        spec_engine.engine.target_cache.seq_len -= gamma
        
        logits = spec_engine.inference(input_ids=verify_tokens)

        count = 0  
        verify_probs = []
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma + 1):
            verify_probs.append(probs[i])
        
        for i, speculation_prob, verify_prob in zip(verify_tokens[0,1:], speculation_probs, verify_probs):
            r = torch.rand(1, device = spec_engine.engine.target.device)
            # print(r, verify_prob[i] / speculation_prob[i])
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                if verbose:
                    spec_stream(i, tokenizer, 'green')
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma - count
                    print("EOS"*20, flush=True)
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

            if tokenizer.eos_token_id == pred_token_idx:
                print("EOS"*20, flush=True)
                break
        
        spec_engine.engine.target_cache.seq_len -= (gamma - count)
        # spec_engine.engine.draft_cache.seq_len -= max(0,(gamma - count - 1))

        if count == gamma:
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            # spec_engine.engine.draft_cache.seq_len -= gamma
            # spec_engine.draft_inference(input_ids=verify_tokens)
            # spec_engine.draft_inference(input_ids=verify_tokens[:,-1:])
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            count += 1

        spec_engine.engine.target_cache.quantize_and_pack()

        next_token = pred_token_idx
    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {spec_engine.engine.target_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    return acceptance_rate, n / (time2 - time1)