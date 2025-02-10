import datasets
import os
import openai
from absl import app, flags
import tenacity
import math
from tqdm import tqdm

flags.DEFINE_integer('start', -1, 'Shard to process')
flags.DEFINE_integer('end', -1, 'Number of shards to process')
flags.DEFINE_string('split', 'train', 'Split to process')
flags.DEFINE_integer('max_examples', 1000000, 'Max examples to process')
flags.DEFINE_integer('save_every', 10000, 'Save every N examples')
flags.DEFINE_string('user', 'Asap7772', 'User to push the dataset to')
FLAGS = flags.FLAGS

PROMPT_LOC_DICT = {
    'backtracking': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/backtracking_v0.txt',
    'is_solution': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/is_solution_v0.txt',
    'verification': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/verification_v0.txt',
    'subgoal_setting': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/subgoal_setting_v0.txt',
    'backward_chaining': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/backward_chaining_v0.txt',
}
PROMPTS = {
    k: open(v).read() for k, v in PROMPT_LOC_DICT.items()
}
for k, v in PROMPTS.items():
    assert '{response}' in v, f'Prompt {k} does not contain {{response}} in {v}'

client = None
@tenacity.retry(stop=tenacity.stop_after_attempt(1000), wait=tenacity.wait_exponential(multiplier=1, min=4, max=7))
def get_response(prompt_type, response, model="meta-llama/Meta-Llama-3.3-70B-Instruct"):
    prompt=PROMPTS[prompt_type].format(response=response)
    output = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return output.choices[0].message.content

def map_fn(examples):
    examples['backtracking_raw'] = [''] * len(examples['text'])
    examples['is_solution_raw'] = [''] * len(examples['text'])
    examples['verification_raw'] = [''] * len(examples['text'])
    for i, response in enumerate(examples['text']):
        examples['backtracking_raw'][i] = get_response('backtracking', response)
        examples['is_solution_raw'][i] = get_response('is_solution', response)
        examples['verification_raw'][i] = get_response('verification', response)
    return examples

def main(_):
    global client
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    ds = datasets.load_dataset('open-web-math/open-web-math', num_proc=os.cpu_count(), split=FLAGS.split)
    
    if FLAGS.max_examples > 0:
        ds = ds.select(range(FLAGS.max_examples))
    
    if FLAGS.start >= 0 and FLAGS.end >= 0 and FLAGS.start < FLAGS.end:
        print('Subsampling the dataset with start={} and end={}'.format(FLAGS.start, FLAGS.end))
        ds = ds.select(range(FLAGS.start, FLAGS.end))

    num_shards = math.ceil(len(ds) / FLAGS.save_every)
    all_ds = []
    for shard_idx in tqdm(range(num_shards), desc='Shards'):
        shard_start = shard_idx * FLAGS.save_every
        shard_end = min((shard_idx + 1) * FLAGS.save_every, len(ds))
        
        curr_shard = ds.select(range(shard_start, shard_end))
        mapped_shard = curr_shard.map(map_fn, batched=True, num_proc=os.cpu_count())
        all_ds.append(mapped_shard)
        
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
    app.run(main)