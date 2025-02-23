import datasets
from transformers import AutoTokenizer
import os
from verl.utils.reward_score.math_eval import MathEvaluator

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

max_tokens = 4096
margin_tokens = 0

ds = datasets.load_dataset('simplescaling/s1K-1.1', split='train')

thinking_key = 'deepseek_thinking_trajectory'
attempt_key = 'deepseek_attempt'
output_path = '/home/anikait.singh/rl_behaviors/cot_datasets/s1_deepseek_sft/positive_control/'
dataset_name = 'Asap7772/s1_deepseek_sft'

# thinking_key = 'gemini_thinking_trajectory'
# attempt_key = 'gemini_attempt'
# output_path = '/home/anikait.singh/rl_behaviors/cot_datasets/s1_gemini_sft/positive_control/'
# dataset_name = 'Asap7772/s1_geminithinking_sft'

os.makedirs(output_path, exist_ok=True)

PROMPT_FORMAT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 5 </answer>.
Assistant: Let me solve this step by step. 
<think>"""

SOLUTION_FORMAT="""<think>
{thinking_text}
</think>
{attempt_text}
<answer>
{extracted_answer}
</answer>
"""

def filter_fn(example):
    curr_query = PROMPT_FORMAT.format(question=example['question'])
    query_tok = tokenizer(curr_query, truncation=True, max_length=max_tokens-margin_tokens)
    query_len = len(query_tok['input_ids'])
    
    if query_len > max_tokens - margin_tokens:
        return False
    return True
ds = ds.filter(filter_fn, num_proc=os.cpu_count())

def map_fn(example):
    curr_query = PROMPT_FORMAT.format(question=example['question'])
    curr_thinking = example[thinking_key]
    curr_attempt = example[attempt_key]
    extracted_answer = MathEvaluator.get_answer_expr(curr_attempt).strip() or example['solution']
    
    curr_completion = SOLUTION_FORMAT.format(thinking_text=curr_thinking, attempt_text=curr_attempt, extracted_answer=extracted_answer)
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
        
        new_query = tokenizer.decode(query_tok['input_ids'], skip_special_tokens=True)
        new_completion = tokenizer.decode(completion_tok['input_ids'], skip_special_tokens=True)
        
        example['question'] = new_query
        example[attempt_key] = new_completion
    
    return example

    
ds = ds.map(map_fn, num_proc=os.cpu_count())
# now change the collumns from question and attempt_key to query and completion
ds = ds.rename_column('question', 'query')
ds = ds.rename_column(attempt_key, 'completion')
other_keys = [key for key in ds.column_names if key not in ['query', 'completion']]
ds = ds.map(lambda x: {'query': x['query'], 'completion': x['completion']}, num_proc=os.cpu_count(), remove_columns=other_keys)

ds = ds.train_test_split(test_size=0.1)
ds.push_to_hub(dataset_name)

ds['train'].to_parquet(f'{output_path}/train.parquet')
ds['test'].to_parquet(f'{output_path}/test.parquet')