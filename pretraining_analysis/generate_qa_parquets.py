import datasets



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

# filter out empty completions and queries
ds = ds.filter(lambda x: len(x['query']) > 0 and len(x['completion']) > 0)

prefix = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step."""

# add prefix to query
ds = ds.map(lambda x: {'query': prefix.format(query=x['query']), 'completion': x['completion']})

# delete all columns except query and completion
ds = ds.remove_columns([col for col in ds.column_names if col not in ['query', 'completion']])

ds_out_name = 'obiwan96/obiwan96open_web_math_qa'
ds.push_to_hub(ds_out_name)
