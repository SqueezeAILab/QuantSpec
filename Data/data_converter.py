import torch
from datasets import load_dataset
import os
from torch.utils.data import TensorDataset
from tqdm import tqdm

def convert_c4_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset

def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_pg19_dataset(tokenizer, seq_len = 4096, end = 20):
    datasetparent = "Data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
    tokenized_prompts = []
    for i in tqdm(range(0,50)):
        prompt = dataset[i]['text']
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:,8000:]
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
        
        for i in range(len(tokenized_prompt)):
             tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
             tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0).repeat(end,1)
    return TensorDataset(data)

def convert_multilexsum_dataset(tokenizer, seq_len = 4096, end = 20):
    shots = 0
    all_data = load_dataset("allenai/multi_lexsum", name="v20230518")
    all_data = all_data.filter(lambda x: x["summary/short"] is not None)

    user_template = "\nYou are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n{context}"
    system_template = "\n\nNow please summarize the case. Summary:"
    all_data = all_data['test']
    all_data = all_data.map(lambda x: {
        "context": '\n\n'.join(x["sources"]),
        "demo": "",
        "answer": x["summary/short"],
        "question": "",
    })
    tokenized_prompts = []
    system_template_len = tokenizer.encode(system_template, return_tensors="pt").shape[1]
    for i in range(50):
        tokenized_prompt = tokenizer.encode(user_template.format(demo="", context=all_data['context'][i]), return_tensors="pt")[:, :seq_len-system_template_len]
        tokenized_prompt[:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        tokenized_prompt = torch.cat((tokenized_prompt, tokenizer.encode(system_template, return_tensors="pt")), axis=1)
        if tokenized_prompt.shape[1] >= seq_len:
            tokenized_prompts.append(tokenized_prompt)
        if len(tokenized_prompts) > 10:
            break
    assert len(tokenized_prompts) > 10, "Less than 10 prompts"
    return tokenized_prompts

def load_infbench(tokenizer, seq_len = 4096, end = 20):
    from datasets import load_dataset, Value, Sequence, Features
    ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"), "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
    data = load_dataset("xinrongzhang2022/infinitebench", features=ft)
   
    # https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py 
    # slightly modified to be consistent with other datasets, shouldn't affect performance        
    user_template = "You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary.\n\n{demo}{context}."
    system_template = "\n\nNow summarize the book. Summary:"
    data = data["longbook_sum_eng"]
    # prompt_template = user_template + "\n\n" + system_template

    def process_example(example):
        update = {"question": example["input"], "demo": ""}
        return update
    
    all_data = data.map(process_example)

    tokenized_prompts = []
    system_template_len = tokenizer.encode(system_template, return_tensors="pt").shape[1]
    for i in range(50):
        tokenized_prompt = tokenizer.encode(user_template.format(demo="", context=all_data[i]['context']), return_tensors="pt")[:, :seq_len-system_template_len]
        tokenized_prompt[:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        tokenized_prompt = torch.cat((tokenized_prompt, tokenizer.encode(system_template, return_tensors="pt")), axis=1)
        if tokenized_prompt.shape[1] >= seq_len:
            tokenized_prompts.append(tokenized_prompt)
        if len(tokenized_prompts) > 10:
            break
    assert len(tokenized_prompts) > 10, "Less than 10 prompts"
    return tokenized_prompts
    # if max_test_samples is not None:
    #     data = data.shuffle(seed=seed).select(range(min(len(data), max_test_samples)))

    # return {
    #     "data": data,
    #     "prompt_template": prompt_template,
    #     "user_template": user_template,
    #     "system_template": system_template,
    #     "post_process": post_process,
    # }


