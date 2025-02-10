import datasets
import os
import openai
from absl import app, flags
import tenacity

flags.DEFINE_integer('shard', 0, 'Shard to process')
flags.DEFINE_integer('num_shards', 1, 'Number of shards to process')
flags.DEFINE_string('split', 'train', 'Split to process')
flags.DEFINE_integer('max_examples', 10000, 'Max examples to process')
FLAGS = flags.FLAGS

PROMPT_LOC_DICT = {
    'backtracking': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/backtracking_v0.txt',
    'is_solution': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/is_solution_v0.txt',
    'verification': '/home/anikait.singh/TinyZero/pretraining_analysis/prompts/verification_v0.txt',
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

def main(_):
    global client
    client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    ds = datasets.load_dataset('open-web-math/open-web-math', num_proc=os.cpu_count(), split=FLAGS.split)
    
    if FLAGS.max_examples > 0:
        ds = ds.select(range(FLAGS.max_examples))
    
    if FLAGS.num_shards > 1:
        print(f'Sharding dataset into {FLAGS.num_shards} shards, processing shard {FLAGS.shard}')
        ds = ds.shard(num_shards=FLAGS.num_shards, index=FLAGS.shard)

    def map_fn(examples):
        examples['backtracking_raw'] = [''] * len(examples['text'])
        examples['is_solution_raw'] = [''] * len(examples['text'])
        examples['verification_raw'] = [''] * len(examples['text'])
        for i, response in enumerate(examples['text']):
            examples['backtracking_raw'][i] = get_response('backtracking', response)
            examples['is_solution_raw'][i] = get_response('is_solution', response)
            examples['verification_raw'][i] = get_response('verification', response)
        return examples

    ds = ds.map(map_fn, batched=True, num_proc=os.cpu_count())
    
    if FLAGS.num_shards > 1:
        ds.push_to_hub(f'Asap7772/open-web-math-processed-{FLAGS.shard+1}-of-{FLAGS.num_shards}')
    else:
        ds.push_to_hub('Asap7772/open-web-math-processed')
    
if __name__ == '__main__':
    app.run(main)