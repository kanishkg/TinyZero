import datasets

ds = datasets.load_dataset("obiwan96/s1-claude-v2")

# rename columns
ds = ds.rename_column("question", "query")
ds = ds.rename_column("answer", "completion")

template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step."""

ds = ds.map(lambda x: {'query': template.format(query=x['query']), 'completion': x['completion'].replace('Let me solve this step by step.', '')})


# make train test split
ds = ds['train']
ds = ds.train_test_split(test_size=0.05)

ds.push_to_hub("obiwan96/s1-claude-sft")
# save as parquet
ds['train'].to_parquet("../ba/data/s1-claude/train.parquet")
ds['test'].to_parquet("../ba/data/s1-claude/test.parquet")