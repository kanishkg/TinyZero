import datasets
import os
import argparse
# output type (query and completion)
parser = argparse.ArgumentParser()
parser.add_argument('--ds_type', type=str, default='backtrack',
                    help='Type of dataset to generate (backtrack, backchain, verification, full, none)')
args = parser.parse_args()

ds_type = args.ds_type
out_keys = ['query', 'completion']
source_dataset = 'Asap7772/open-web-math-processed-v2'
test_per = 0.1
ds = datasets.load_dataset(source_dataset, split='train')

if ds_type == 'backtrack':
    filtered_ds_backtrack = ds.filter(lambda x: x['is_backtrack'] and x['is_backtrack'].lower() == 'yes', num_proc=os.cpu_count())
    filtered_ds_backtrack.train_test_split(test_size=test_per)
    filtered_ds_backtrack.push_to_hub(f'Asap7772/open-web-math-{ds_type}-processed-v2')
    print(f"Pushed {ds_type} dataset to hub.")
elif ds_type == 'backchain':
    filtered_ds_backchain = ds.filter(lambda x: x['is_backchain'] and x['is_backchain'].lower() == 'yes', num_proc=os.cpu_count())
    filtered_ds_backchain.train_test_split(test_size=test_per)
    filtered_ds_backchain.push_to_hub(f'Asap7772/open-web-math-{ds_type}-processed-v2')
    print(f"Pushed {ds_type} dataset to hub.")
elif ds_type == 'verification':
    filtered_ds_verification = ds.filter(lambda x: x['is_verification'] and x['is_verification'].lower() == 'yes', num_proc=os.cpu_count())
    filtered_ds_verification.train_test_split(test_size=test_per)
    filtered_ds_verification.push_to_hub(f'Asap7772/open-web-math-{ds_type}-processed-v2')
    print(f"Pushed {ds_type} dataset to hub.")
elif ds_type == 'full':
    def filter_fn(x):
        return x['is_backtrack'] and x['is_backtrack'].lower() == 'yes' or x['is_backchain'] and x['is_backchain'].lower() == 'yes' or x['is_verification'] and x['is_verification'].lower() == 'yes'
    filtered_ds_full = ds.filter(filter_fn, num_proc=os.cpu_count())
    filtered_ds_full.train_test_split(test_size=test_per)
    filtered_ds_full.push_to_hub(f'Asap7772/open-web-math-{ds_type}-processed-v2')
    print(f"Pushed {ds_type} dataset to hub.")
elif ds_type == 'none':
    def filter_fn(x):
        curr = x['is_backtrack'] and x['is_backtrack'].lower() == 'yes' or x['is_backchain'] and x['is_backchain'].lower() == 'yes' or x['is_verification'] and x['is_verification'].lower() == 'yes'
        return not curr
    filtered_ds_none = ds.filter(filter_fn, num_proc=os.cpu_count())
    filtered_ds_none.train_test_split(test_size=test_per)
    filtered_ds_none.push_to_hub(f'Asap7772/open-web-math-{ds_type}-processed-v2')
    print(f"Pushed {ds_type} dataset to hub.")
else:
    raise ValueError(f"Invalid dataset type: {ds_type}")

