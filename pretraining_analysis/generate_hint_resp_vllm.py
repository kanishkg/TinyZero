import os
os.system('export VLLM_WORKER_MULTIPROC_METHOD=spawn')
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import re
import datasets
from collections import defaultdict

from verl.utils.reward_score.math_eval import MathEvaluator

def is_equiv(gold, answer) -> bool:
    return MathEvaluator.is_correct_sync(gold, answer)

def get_answer_boxed(s):
    try:
        return MathEvaluator.get_answer_expr(s)
    except Exception as e:
        return ''

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-32B-Instruct", help='Model name')
parser.add_argument('--dataset_name', type=str, default='metamath-filtered', help='Dataset name')
parser.add_argument('--output_name',  type=str, default='metamath-filtered-qwen-gen', help='Output name')
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--max_prompt_length', type=int, default=1024, help='Max prompts to process')
parser.add_argument('--max_length', type=int, default=4096, help='Max length of the input')
parser.add_argument('--temperature', type=float, default=2.0, help='Sampling temperature')
parser.add_argument('--top_p', type=int, default=1.0, help='Top k sampling')
parser.add_argument('--min_p', type=float, default=0.3, help='Top p sampling')
parser.add_argument('--user', type=str, default='Asap7772', help='User to push the dataset to')
parser.add_argument('--n', type=int, default=16, help='Number of examples to generate')

def get_prompts(ds, tokenizer, max_length=4096, max_prompt_length=1024):
    prompts, answers, samples = [], [], []
    tokenized_inputs = tokenizer(ds['problem'])
    for e, example in tqdm(enumerate(tokenized_inputs['input_ids']), desc="Truncating prompts"):
        if len(example) > max_length-max_prompt_length:
            sample = tokenizer.decode(example[: max_length - max_prompt_length])
            sample = sample[: sample.rfind("\n")]
            samples += [sample]
        else:
            samples += [ds['problem'][e]]
        answers += [ds['answer'][e]]

    for example in tqdm(samples, desc="Generating prompts"):
        prompt = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
            {"role": "user", "content": example}
        ]
        prompts += [prompt]
  
    new_prompts = [tokenizer.apply_chat_template(p,tokenize=False, add_generation_prompt=True) for p in prompts]
    return new_prompts, answers

def main(args):
    if args.dataset_name == 'metamath-filtered':
        ds = datasets.load_dataset('active-reasoning/MetaMATH-filtered', num_proc=os.cpu_count()-2, split=args.split)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
        
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))
    
    # filter examples where 'contain_problem' is no or 'contain_solution' is no
    if args.dataset_name == 'openwebmath' or args.dataset_name == 'openwebmath_backtrack':
        ds = ds.filter(lambda x: x['contain_problem'] != 'no' and x['contain_solution'] != 'no')
        print(f"Number of examples after filtering: {len(ds)}")

    llm = LLM(
        model=args.model_name,
        tokenizer_mode="auto",
        max_num_seqs=64,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    num_batches = max(math.ceil(len(ds) / args.save_every), 1)
    print(f'Number of batches: {num_batches}')
    print(f'Number of examples: {len(ds)}')
    
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts, answers = get_prompts(curr_batch, tokenizer, max_length=args.max_length, max_prompt_length=args.max_prompt_length)
        sampling_params = SamplingParams(
            max_tokens=args.max_length,
            top_p=args.top_p,
            min_p=args.min_p,
            temperature=args.temperature,
            n=args.n,
        )

        responses = llm.generate(prompts, sampling_params=sampling_params)
        
        outputs_dict = defaultdict(list)
        for i, response in enumerate(responses):
            outputs, outputs_answer, outputs_correct = [], [], []
            for j in range(args.n):
                output = response.outputs[j].text.strip()
                output_answer = get_answer_boxed(output)
                target_answer = answers[i]
                output_correct = is_equiv(target_answer, output_answer)
                
                outputs.append(output)
                outputs_answer.append(output_answer)
                outputs_correct.append(output_correct)
            
            outputs_success_rate = sum(outputs_correct) / len(outputs_correct)
            outputs_dict['completion'].append(outputs)
            outputs_dict['completion_answer'].append(outputs_answer)
            outputs_dict['completion_correct'].append(outputs_correct)
            outputs_dict['completion_succ_rate'].append(outputs_success_rate)
        
        curr_batch = curr_batch.add_column('completion', outputs_dict['completion'])
        curr_batch = curr_batch.add_column('completion_answer', outputs_dict['completion_answer'])
        curr_batch = curr_batch.add_column('completion_correct', outputs_dict['completion_correct'])
        curr_batch = curr_batch.add_column('completion_succ_rate', outputs_dict['completion_succ_rate'])

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'{args.user}/{args.output_name}_{suffix}'
            ds_so_far.push_to_hub(ds_out_name)
        except Exception as e:
            print(f'Error saving dataset: {e}')
            continue
    
    try:
        ds_so_far = datasets.concatenate_datasets(all_ds)
        if args.start >= 0 and args.end >= 0 and args.start < args.end:
            suffix = f'_{args.start}_{args.end}'
        else:
            suffix = ''
        ds_out_name = f'{args.user}/{args.output_name}_{suffix}'
        ds_so_far.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
