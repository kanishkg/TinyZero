import os
import numpy as np
import datasets
from transformers import AutoTokenizer



# data_names = [
#     'obiwan96/obiwan96open_web_math_qav2_0_11616', 
#     'obiwan96/obiwan96open_web_math_qav2_11616_23232',
#     'obiwan96/obiwan96open_web_math_qav2_23232_34848',
#     'obiwan96/obiwan96open_web_math_qav2_34848_46467',
# ]
data_names = [
    # 'obiwan96/obiwan96open_web_math_qav2_none_0_100000',
    # 'obiwan96/obiwan96open_web_math_qav2_none_100000_200000',
    # 'obiwan96/obiwan96open_web_math_qav2_none_200000_300000',
    # 'obiwan96/obiwan96open_web_math_qav2_none_300000_400000',
    
    'Asap7772/Asap7772open_web_math_qav3_68750_75000',
    'Asap7772/Asap7772open_web_math_qav3_93750_100000',
    'Asap7772/Asap7772open_web_math_qav3_56250_62500',
    'Asap7772/Asap7772open_web_math_qav3_50000_56250',
    'Asap7772/Asap7772open_web_math_qav3_62500_68750',
    'Asap7772/Asap7772open_web_math_qav3_81250_87500',
    'Asap7772/Asap7772open_web_math_qav3_87500_93750',
    'Asap7772/Asap7772open_web_math_qav3_75000_81250',
    'Asap7772/Asap7772open_web_math_qav3_575000_600000',
    'Asap7772/Asap7772open_web_math_qav3_450000_475000',
    'Asap7772/Asap7772open_web_math_qav3_475000_500000',
    'Asap7772/Asap7772open_web_math_qav3_400000_425000',
    'Asap7772/Asap7772open_web_math_qav3_550000_575000',
    'Asap7772/Asap7772open_web_math_qav3_425000_450000',
    'Asap7772/Asap7772open_web_math_qav3_500000_525000',
    'Asap7772/Asap7772open_web_math_qav3_525000_550000',
    'obiwan96/obiwan96open_web_math_qav3_300000_350000',
    'obiwan96/obiwan96open_web_math_qav3_350000_400000',
    'obiwan96/obiwan96open_web_math_qav3_250000_300000',
    'obiwan96/obiwan96open_web_math_qav3_200000_250000',
    'obiwan96/obiwan96open_web_math_qav3_150000_200000',
    'obiwan96/obiwan96open_web_math_qav3_0_50000',
    'obiwan96/obiwan96open_web_math_qav3_100000_150000',

]
all_ds = []
for data_name in data_names:
    ds = datasets.load_dataset(data_name)
    # use train split
    ds = ds['train']
    all_ds.append(ds)


ds = datasets.concatenate_datasets(all_ds)

# filter out empty completions and queries
print(f"Number of examples: {len(ds)}")
ds = ds.filter(lambda x: len(x['query']) > 0 and len(x['completion']) > 0)
print(f"Number of examples: {len(ds)}")

prefix = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step."""

# add prefix to query
ds = ds.map(lambda x: {'query': prefix.format(query=x['query']), 'completion': '\n'+x['completion']})

# delete all columns except query and completion
ds = ds.remove_columns([col for col in ds.column_names if col not in ['query', 'completion']])

# create train and validation splits
ds = ds.train_test_split(test_size=0.05)

train_completion = ds['train']['completion']
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tokens = tokenizer(train_completion)
lens = [len(t) for t in tokens['input_ids']]
print(f"Max length: {max(lens)}")
print(f"Min length: {min(lens)}")
print(f"Mean length: {np.mean(lens)}")
# target_len = 8300000
# cumsum = 0
# keep_idx = []
# for i, l in enumerate(lens):
#     # clip l at 4096
#     l = min(l, 4096)
#     if cumsum + l <= target_len:
#         cumsum += l
#         keep_idx.append(i)
#     else:
#         break

# ds['train'] = ds['train'].select(keep_idx)
# print(f"Kept {len(keep_idx)} examples with total {cumsum} tokens")

ds_out_name = 'obiwan96/obiwan96open_web_math_qav3_none'
ds.push_to_hub(ds_out_name)

# save as train.parquet and test.parquet
if not os.path.exists('/home/kanishk/ba/owm_mathv3'):
    os.makedirs('/home/kanishk/ba/owm_mathv3')
ds['train'].to_parquet('/home/kanishk/ba/owm_mathv3/train.parquet')
ds['test'].to_parquet('/home/kanishk/ba/owm_mathv3/test.parquet')
