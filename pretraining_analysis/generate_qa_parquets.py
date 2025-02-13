import datasets

def filter_fn(example):
    

data_names = [
    'obiwan96/obiwan96open_web_math_qa_0_11616', 
    'obiwan96/obiwan96open_web_math_qa_11616_23232',
    'obiwan96/obiwan96open_web_math_qa_23232_34848',
    'obiwan96/obiwan96open_web_math_qa_34848_46467',
]

all_ds = []
for data_name in data_names:
    ds = datasets.load_dataset(data_name)
    all_ds.append(ds)

ds = datasets.concatenate_datasets(all_ds)

ds_out_name = 'obiwan96/obiwan96open_web_math_qa'
ds.push_to_hub(ds_out_name)
