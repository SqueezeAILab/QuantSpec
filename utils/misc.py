from sympy import symbols, Eq, solve
from termcolor import colored
import random

def spec_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

def log_csv(file_path, header, entry):
    try:
        with open(file_path, 'r') as f:
            contents = f.read()
    except FileNotFoundError:
        contents = ""

    if not contents:
        with open(file_path, 'a') as f:
            f.write(header)
    
    with open(file_path, 'a') as f:
        f.write(entry)

def print_config(draft, target, dataset, prefill, gen_len, gamma, top_k, top_p, temperature, quantized_draft, k_bits, v_bits, group_size, residual_length):
    print(colored("####################################### Config #######################################", 'blue'), flush=True)
    print(colored(f"Dataset: {dataset}", 'blue'), flush=True)
    print(colored(f"Draft: {draft.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Target: {target.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Prefill Length: {prefill}", 'blue'), flush=True)
    print(colored(f"Generation Length: {gen_len}", 'blue'), flush=True)
    print(colored(f"Gamma: {gamma}", 'blue'), flush=True)
    print(colored(f"Sampling Method: top_k = {top_k}, top_p = {top_p}, temperature = {temperature}", 'blue'), flush=True)
    print(colored(f"Quantized Draft: {quantized_draft}", 'blue'), flush=True)
    print(colored(f"K Bits: {k_bits}", 'blue'), flush=True)
    print(colored(f"V Bits: {v_bits}", 'blue'), flush=True)
    print(colored(f"Group Size: {group_size}", 'blue'), flush=True)
    print(colored(f"Residual Length: {residual_length}", 'blue'), flush=True)
    print(colored("######################################################################################\n", 'blue'), flush=True)

