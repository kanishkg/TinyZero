import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--user', type=str, default='Asap7772', help='User to push the dataset to')

def get_prompts(ds, tokenizer, prompt_templates):
    prompts = []
    tokenized_inputs = tokenizer(ds['text'])
    samples = []
    max_seq_length = 4096
    for e, example in tqdm(enumerate(tokenized_inputs['input_ids']), desc="Truncating prompts"):
        if len(example) > max_seq_length-1024:
            sample = tokenizer.decode(example[: max_seq_length - 1024])
            sample = sample[: sample.rfind("\n")]
            samples += [sample]
        else:
            samples += [ds['text'][e]]

    for example in tqdm(samples, desc="Generating prompts"):
        prompt = prompt_templates['qa'].format(response=example)
        prompt = [{'role': 'user', 'content': prompt}]
        prompts += [prompt]
  
    new_prompts = [tokenizer.apply_chat_template(
        p,
        tokenize=False,
    ) for p in prompts]
    return new_prompts

def parse_output(output):
    query_match = re.search(r'<question>(.*?)</question>', output, re.DOTALL)
    think_match = re.search(r'<thoughts>(.*?)</thoughts>', output, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    
    query = query_match.group(1) if query_match else ""
    think = think_match.group(1) if think_match else ""
    answer = answer_match.group(1) if answer_match else ""
    
    completion = f"<think>{think}</think>\n<answer>{answer}</answer>" if think or answer else ""
    return query, completion

def main(args):
    prompt_templates = {
        'qa': """Your goal is to split the text into a question, thought and an answer.

Make sure that the question is in the text. 
Make sure that the answer and the process of the writer to get to the answer are in the text. Do not change the wording of either.
Your goal is to copy and paste the question, the process and the answer into the correct sections.

Write the question in <question>...</question>.
For the answer, split the answer into the process towards reaching the answer and the final answer.
Write the process in <thoughts>thinking process of the writer</thoughts> and the final answer in <answer>...</answer>.

Directly copy, do not change the text.
Here is the text:
<text>
{response}
</text>"""
}
    for k, v in prompt_templates.items():
        assert '{response}' in v, f'Prompt {k} does not contain {{response}} in {v}'

    ds = datasets.load_dataset('Asap7772/open-web-math-backtrack-processed-v2', num_proc=os.cpu_count()-2, split=args.split)
        
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))
    llm = LLM(
        model='Qwen/Qwen2.5-32B-Instruct',
        tokenizer_mode="auto",
        max_num_seqs=64,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    num_batches = math.ceil(len(ds) / args.save_every)
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, tokenizer, prompt_templates)
        sampling_params = SamplingParams(
            max_tokens=4096+1024,
            temperature=0,
        )

        responses = llm.generate(prompts, sampling_params=sampling_params)

        outputs_dict = {
            'query': [None] * len(curr_batch),
            'completion': [None] * len(curr_batch)
        }
        
        for i, response in enumerate(responses):
            output = response.outputs[0].text.strip()
            query, completion = parse_output(output)
            outputs_dict['query'][i] = query
            outputs_dict['completion'][i] = completion
        
        curr_batch = curr_batch.add_column('query', outputs_dict['query'])
        curr_batch = curr_batch.add_column('completion', outputs_dict['completion'])

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'{args.user}open_web_math_qa{suffix}'
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
        ds_out_name = f'{args.user}open_web_math_qa{suffix}'
        ds_so_far.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
