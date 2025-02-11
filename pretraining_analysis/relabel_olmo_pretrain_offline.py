import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams


flags.DEFINE_integer('start', -1, 'Shard to process')
flags.DEFINE_integer('end', -1, 'Number of shards to process')
flags.DEFINE_string('split', 'train', 'Split to process')
flags.DEFINE_integer('max_examples', 1000000, 'Max examples to process')
flags.DEFINE_integer('save_every', 10000, 'Save every N examples')
flags.DEFINE_string('user', 'Asap7772', 'User to push the dataset to')
FLAGS = flags.FLAGS

PROMPT_LOC_DICT = {
    'backtracking': './pretraining_analysis/prompts/backtracking_v0.txt',
    'is_solution': './pretraining_analysis/prompts/is_solution_v0.txt',
    'verification': './pretraining_analysis/prompts/verification_v0.txt',
    'subgoal_setting': './pretraining_analysis/prompts/subgoal_setting_v0.txt',
    'backward_chaining': './pretraining_analysis/prompts/backward_chaining_v0.txt',
}


def get_prompts(ds, prompt_templates):
    prompts = []
    for example in tqdm(ds['text'], desc="Generating prompts"):
        backtracking_prompt = prompt_templates['backtracking'].format(response=example)
        is_solution_prompt = prompt_templates['is_solution'].format(response=example)
        verification_prompt = prompt_templates['verification'].format(response=example)
        subgoal_setting_prompt = prompt_templates['subgoal_setting'].format(response=example)
        backward_chaining_prompt = prompt_templates['backward_chaining'].format(response=example)
        prompts += [backtracking_prompt, is_solution_prompt, verification_prompt, subgoal_setting_prompt, backward_chaining_prompt]
    return prompts

def main():
    prompt_templates = {
            k: open(v).read() for k, v in PROMPT_LOC_DICT.items()
        }
    for k, v in prompt_templates.items():
        assert '{response}' in v, f'Prompt {k} does not contain {{response}} in {v}'

    ds = datasets.load_dataset('open-web-math/open-web-math', num_proc=os.cpu_count()-2, split=FLAGS.split)
        
    if FLAGS.max_examples > 0:
        ds = ds.select(range(FLAGS.max_examples))
    
    if FLAGS.start >= 0 and FLAGS.end >= 0 and FLAGS.start < FLAGS.end:
        print('Subsampling the dataset with start={} and end={}'.format(FLAGS.start, FLAGS.end))
        ds = ds.select(range(FLAGS.start, FLAGS.end))
    llm = LLM(
        model='meta-llama/Meta-Llama-3.3-70B-Instruct',
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
    )

    num_batches = math.ceil(len(ds) / FLAGS.save_every)
    batch_size = FLAGS.save_every
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * FLAGS.save_every
        batch_end = min((shard_idx + 1) * FLAGS.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, prompt_templates)

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
            if FLAGS.start >= 0 and FLAGS.end >= 0 and FLAGS.start < FLAGS.end:
                suffix = f'_{FLAGS.start}_{FLAGS.end}'
            else:
                suffix = ''
            ds_out_name = f'{FLAGS.user}open_web_math_raw{suffix}'
            ds_so_far.push_to_hub(ds_out_name)
        except Exception as e:
            print(f'Error saving dataset: {e}')
            continue
    
    try:
        ds_so_far = datasets.concatenate_datasets(all_ds)
        if FLAGS.start >= 0 and FLAGS.end >= 0 and FLAGS.start < FLAGS.end:
            suffix = f'_{FLAGS.start}_{FLAGS.end}'
        else:
            suffix = ''
        ds_out_name = f'{FLAGS.user}open_web_math_raw{suffix}'
        ds.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':
    main()
