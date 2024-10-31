from datasets import load_dataset
from tqdm import tqdm
import secrets
import random
import torch
import json
import os

def build_chat_input_lwm(tokenizer, message, prefill=127*1024):
    # chat format:
    # single-turn: You are a helpful assistant. USER: {} \n ASSISTANT:
    book = tokenizer.encode(message)[:prefill-84]
    prompt = "You are a helpful assistant. USER: Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + tokenizer.decode(book, skip_special_tokens=True) + "\n[end of the book]\n\nNow you have read it. Please summarize it for me. First, tell me the title and the author, and then tell the story in 400 words.\n\nASSISTANT: "
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    return input_tokens

def get_dataset(dataset_name, tokenizer=None, datalen=None, task=None):
    if dataset_name == '128k':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(len(dataset))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'gs':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(20)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'one-shot':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(1)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts

    elif dataset_name == 'demo':
        dataset = load_dataset("narrativeqa", cache_dir='/rscratch/rishabhtiwari/cache/')
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[2]]['document']['text'][3:1024*500])
        tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'lwm':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        for i in range(20):
            tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[i]]['document']['text'][3:1024*500])
            if tokenized_prompt.shape[-1] != 127*1024:
                print(i, tokenized_prompt.shape)
                continue
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts
    
    elif dataset_name == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir='/rscratch/rishabhtiwari/cache/')
        tokenized_prompts = []
        tokenized_prompts.append(tokenizer.encode("\n\n".join(dataset['text']), return_tensors="pt"))
        return tokenized_prompts
    
    elif dataset_name == "c4":
        tokenized_prompts = []
        dataset = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            use_auth_token=False,
            cache_dir='/rscratch/rishabhtiwari/cache/'
        )
        tokenized_prompts.append(tokenizer.encode(" ".join(dataset[0:1100]['text']), return_tensors="pt")[:, :(256*datalen)])
        return tokenized_prompts
    
    elif dataset_name == 'multi_lexsum':
        shots = 0
        all_data = load_dataset("allenai/multi_lexsum", name="v20230518", cache_dir='/rscratch/rishabhtiwari/cache/')
        all_data = all_data.filter(lambda x: x["summary/short"] is not None)

        user_template = "You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n{context}"
        system_template = "\n\nNow please summarize the case. Summary:"
        all_data = all_data['test']
        all_data = all_data.map(lambda x: {
            "context": '\n\n'.join(x["sources"]),
            "demo": "",
            "answer": x["summary/short"],
            "question": "",
        })
        tokenized_prompts = []
        system_template_len = len(tokenizer.encode(system_template, return_tensors="pt"))
        for i in range(20):
            tokenized_prompt = tokenizer.encode(user_template.format(demo="", context=all_data['context'][i]), return_tensors="pt")[:datalen-system_template_len]
            tokenized_prompt = torch.cat((tokenized_prompt, tokenizer.encode(system_template, return_tensors="pt")), axis=1)
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts
    
    else:
        raise Exception("Dataset not found")