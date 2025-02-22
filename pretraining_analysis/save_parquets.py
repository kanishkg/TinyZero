import datasets
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

max_tokens = 4096
margin_tokens = 0

ds = datasets.load_dataset('obiwan96/obiwan96open_web_math_qav3_none')

def filter_fn(example):
    curr_query = example['query']
    query_tok = tokenizer(curr_query, truncation=True, max_length=max_tokens-margin_tokens)
    query_len = len(query_tok['input_ids'])
    
    if query_len > max_tokens - margin_tokens:
        return False
    return True
ds = ds.filter(filter_fn, num_proc=os.cpu_count())

def map_fn(example):
    curr_query = example['query']
    curr_completion = example['completion']
    query_tok = tokenizer(curr_query, truncation=True, max_length=max_tokens-margin_tokens)
    completion_tok = tokenizer(curr_completion, truncation=True, max_length=max_tokens-margin_tokens)
    total_len = len(query_tok['input_ids']) + len(completion_tok['input_ids'])
    
    if total_len > max_tokens:
        len_query = len(query_tok['input_ids'])
        len_completion = len(completion_tok['input_ids'])
        if len_query > max_tokens - margin_tokens:
            len_query = max_tokens - margin_tokens
            len_completion = 0
        else:
            len_completion = max_tokens - margin_tokens - len_query
        query_tok['input_ids'] = query_tok['input_ids'][:len_query]
        query_tok['attention_mask'] = query_tok['attention_mask'][:len_query]
        completion_tok['input_ids'] = completion_tok['input_ids'][:len_completion]
        completion_tok['attention_mask'] = completion_tok['attention_mask'][:len_completion]
        
        new_query = tokenizer.decode(query_tok['input_ids'])
        new_completion = tokenizer.decode(completion_tok['input_ids'])
        
        example['query'] = new_query
        example['completion'] = new_completion
    
    return example
    
ds = ds.map(map_fn, num_proc=os.cpu_count())
ds.push_to_hub('Asap7772/obiwan96open_web_math_qav3_none')

ds['train'].to_parquet('/home/anikait.singh/rl_behaviors/cot_datasets/data_math_qv3/method/train.parquet')
ds['test'].to_parquet('/home/anikait.singh/rl_behaviors/cot_datasets/data_math_qv3/method/test.parquet')