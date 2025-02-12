import datasets
import os
import argparse
# output type (query and completion)
parser = argparse.ArgumentParser()
parser.add_argument('--ds_type', type=str, default='backtrack',
                    help='Type of dataset to generate (backtrack, backchain, verification, full, none)')
parser.add_argument('--num_samples', type=int, default=40000,)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

ds_type = args.ds_type
samples = args.num_samples
out_keys = ['query', 'completion']
source_dataset = 'Asap7772/open-web-math-processed-v2'
output_path = '/home/anikait.singh/rl_behaviors/cot_datasets/pretrained_data/'
test_per = 0.1
ds = datasets.load_dataset(source_dataset, split='train')

if ds_type == 'backtrack':
    filtered_ds = ds.filter(lambda x: x['is_backtrack'] and x['is_backtrack'].lower() == 'yes', num_proc=os.cpu_count())
elif ds_type == 'baseline':
    filtered_ds = ds
elif ds_type == 'method':
    filtered_ds_backtrack = ds.filter(lambda x: x['is_backtrack'] and x['is_backtrack'].lower() == 'yes', num_proc=os.cpu_count())
    filtered_ds_backchain = ds.filter(lambda x: x['is_backchain'] and x['is_backchain'].lower() == 'yes', num_proc=os.cpu_count())
    filtered_ds_verification = ds.filter(lambda x: x['is_verification'] and x['is_verification'].lower() == 'yes', num_proc=os.cpu_count())
    
    min_samples = min(len(filtered_ds_backtrack), len(filtered_ds_backchain), len(filtered_ds_verification))
    filtered_ds_backtrack = filtered_ds_backtrack.select(range(min_samples))
    filtered_ds_backchain = filtered_ds_backchain.select(range(min_samples))
    filtered_ds_verification = filtered_ds_verification.select(range(min_samples))
    filtered_ds = datasets.concatenate_datasets([filtered_ds_backtrack, filtered_ds_backchain, filtered_ds_verification])
elif ds_type == 'negative':
    def filter_fn(x):
        curr = x['is_backtrack'] and x['is_backtrack'].lower() == 'yes' or x['is_backchain'] and x['is_backchain'].lower() == 'yes' or x['is_verification'] and x['is_verification'].lower() == 'yes'
        return not curr
    filtered_ds = ds.filter(filter_fn, num_proc=os.cpu_count())
else:
    raise ValueError(f"Invalid dataset type: {ds_type}")

def map_fn(examples):
    return_dict = {'query': [], 'completion': []}
    for text in examples['text']:
        return_dict['query'].append('')
        return_dict['completion'].append(text)
    return return_dict

filtered_ds = filtered_ds.map(map_fn, batched=True, num_proc=os.cpu_count(), remove_columns=filtered_ds.column_names)
assert 'query' in filtered_ds.column_names and 'completion' in filtered_ds.column_names, f"Columns not found in dataset: {filtered_ds.column_names}"

filtered_ds = filtered_ds.shuffle(seed=args.seed)
test_samples = int(samples * test_per)
total_samples = samples + test_samples
filtered_ds = filtered_ds.select(range(total_samples))
train_ds = filtered_ds.select(range(samples))
test_ds = filtered_ds.select(range(samples, total_samples))

output_path_train = os.path.join(output_path, ds_type, 'train.parquet')
output_path_test = os.path.join(output_path, ds_type, 'test.parquet')
os.makedirs(os.path.dirname(output_path_train), exist_ok=True)
train_ds.to_parquet(output_path_train)
test_ds.to_parquet(output_path_test)

print("Saved datasets to:", output_path)
print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
