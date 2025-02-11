import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=1000000, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--user', type=str, default='Asap7772', help='User to push the dataset to')

PROMPT_LOC_DICT = {
    'backtracking': './pretraining_analysis/prompts/backtracking_v0.txt',
    'is_solution': './pretraining_analysis/prompts/is_solution_v0.txt',
    'verification': './pretraining_analysis/prompts/verification_v0.txt',
    'subgoal_setting': './pretraining_analysis/prompts/subgoal_setting_v0.txt',
    'backward_chaining': './pretraining_analysis/prompts/backward_chaining_v0.txt',
}


def get_prompts(ds, tokenizer, prompt_templates):
    prompts = []
    print(len(ds['text']))
    import pdb; pdb.set_trace()
    for example in tqdm(ds['text'], desc="Generating prompts"):
        backtracking_prompt = prompt_templates['backtracking'].format(response=example)
        backtracking_prompt = [{'role': 'user', 'content': backtracking_prompt}]
        is_solution_prompt = prompt_templates['is_solution'].format(response=example)
        is_solution_prompt = [{'role': 'user', 'content': is_solution_prompt}]
        verification_prompt = prompt_templates['verification'].format(response=example)
        verification_prompt = [{'role': 'user', 'content': verification_prompt}]
        subgoal_setting_prompt = prompt_templates['subgoal_setting'].format(response=example)
        subgoal_setting_prompt = [{'role': 'user', 'content': subgoal_setting_prompt}]
        backward_chaining_prompt = prompt_templates['backward_chaining'].format(response=example)
        backward_chaining_prompt = [{'role': 'user', 'content': backward_chaining_prompt}]
        prompts += [backtracking_prompt, is_solution_prompt, verification_prompt, subgoal_setting_prompt, backward_chaining_prompt]
    new_prompts = [tokenizer.apply_chat_template(
        p,
        tokenize=False,
    ) for p in prompts]

    return new_prompts

def main(args):
    prompt_templates = {
            k: open(v).read() for k, v in PROMPT_LOC_DICT.items()
        }
    for k, v in prompt_templates.items():
        assert '{response}' in v, f'Prompt {k} does not contain {{response}} in {v}'

    ds = datasets.load_dataset('open-web-math/open-web-math', num_proc=os.cpu_count()-2, split=args.split)
        
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))
    llm = LLM(
        model='meta-llama/Llama-3.3-70B-Instruct',
        tokenizer_mode="auto",
        max_num_seqs=32,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

    num_batches = math.ceil(len(ds) / args.save_every)
    batch_size = args.save_every
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, prompt_templates, tokenizer)

        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0,
        )
        responses = llm.generate(prompts, sampling_params=sampling_params)
        curr_batch['backtracking_raw'] = [''] * len(responses)
        curr_batch['is_solution_raw'] = [''] * len(responses)
        curr_batch['verification_raw'] = [''] * len(responses)
        curr_batch['subgoal_setting_raw'] = [''] * len(responses)
        curr_batch['backward_chaining_raw'] = [''] * len(responses)

        for i, response in enumerate(responses):
            output = response.outputs[0].text.strip()
            idx = i % 5
            if idx == 0:
                curr_batch['backtracking_raw'][i // 5] = output
            elif idx == 1:
                curr_batch['is_solution_raw'][i // 5] = output
            elif idx == 2:
                curr_batch['verification_raw'][i // 5] = output
            elif idx == 3:
                curr_batch['subgoal_setting_raw'][i // 5] = output
            elif idx == 4:
                curr_batch['backward_chaining_raw'][i // 5] = output

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'{args.user}open_web_math_raw{suffix}'
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
        ds_out_name = f'{args.user}open_web_math_raw{suffix}'
        ds.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
