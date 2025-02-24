import json
from tqdm import tqdm

import datasets
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request



PROMPT= """You are a reasoning agent that is tasked with solving a problem.
Reasoning Trajectory Template

Write your thoughts in <think> </think> tags.
The answer should be written in <answer> </answer> tags.

Requirements for the answer:
- Use only valid operations and starting elements
- Show clear steps from initial state to goal state
- Show reasoning operations like verification, backtracking, sub-goal setting, working backwards from the goal

Problem-solving principles to follow:
- Verify your work explicitly whenever you can and backtrack when needed
    For example, use phrases like:
    - "Let me check if my answer is correct"
    - "This doesn't seem right, let me try something else"
    - "Let me check my work"
    - "Let's substitute the value of x into the equation"
- Break down complex goals into manageable sub-goals
    - Use phrases like:
    - "Let me break this down into smaller steps"
    - "To solve this problem, I first need to..."
- If stuck, return to a previous state and try a different approach
    - Use phrases like:
    - "Let me go back and correct my previous step"
    - "Let me go back and try a different approach"
Format for writing solutions:
<think> thoughts here </think>
<answer> answer </answer>

Remember:
- It is necessary to show your work in <think> </think> tags.
- It is required that you makes some mistakes in your reasoning and verify your work and then backtrack to correct your mistakes.
- It is required that you break down complex goals into manageable sub-goals.
- It is required that you verify your work explicitly whenever you can and backtrack when needed.
- Only write the final answer in <answer> </answer> tags.
- You answer should always have mistakes, verification, and backtracking.
Use up to 1000 words.
"""
USER_PROMPT = "Solve the following problem: {question}"

client = anthropic.Anthropic()

s1ds = datasets.load_dataset("simplescaling/s1K")

requests = []
questions = []
for s, sample in enumerate(s1ds["train"]):
    question = sample["question"]
    questions.append(question)
    user_message = USER_PROMPT.format(question=question)
    requests.append(Request(
                custom_id=f"s1-cot-{s}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2048,
                    temperature=0.7,
                    system=[
                        {
                            "type": "text",
                            "text": PROMPT,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ]
                )
            ))


request_map = {
        req['custom_id']: {
                "prompt": req['params']['messages'][0]['content'],
                "system": req['params']['system'][0]['text'],
            }
            for req in requests
        }
        
answers = []
message_batch = client.messages.batches.create(requests=requests)
batch_id = message_batch.id
print(f"Batch {batch_id} submitted successfully")

from datetime import datetime, timedelta
import time
start_time = datetime.now()
end_time = start_time + timedelta(hours=24)
poll_interval = 60  # Start with 1 minute interval

print(f"Starting to poll batch {batch_id}")
print(f"Will poll until {end_time} or batch completion")

while datetime.now() < end_time:
    try:
        batch_status = client.messages.batches.retrieve(batch_id)
        
        counts = batch_status.request_counts
        total = sum([counts.processing, counts.succeeded, counts.errored, 
                    counts.canceled, counts.expired])
        processed = counts.succeeded + counts.errored

        if counts.errored > 0:
            breakpoint()
        
        print(f"Batch {batch_id} status: {batch_status.processing_status}")
        print(f"Status: {processed}/{total} requests processed")
        print(f"Succeeded: {counts.succeeded}, Errored: {counts.errored}, "
                f"Canceled: {counts.canceled}, Expired: {counts.expired}")
        
        if batch_status.processing_status == "ended":
            print("Batch processing completed!")
            break
        
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 120)
        
    except Exception as e:
        print(f"Error polling batch status: {str(e)}")
        time.sleep(poll_interval)

answers = []
for r,result in enumerate(client.messages.batches.results(batch_id)):
    if result.result.type == "succeeded":
        prompts = request_map[result.custom_id]
        answers.append(result.result.message.content[0].text)
        
new_dataset = datasets.Dataset.from_dict({"question": questions, "answer": answers})
new_dataset.push_to_hub("obiwan96/s1-claude-v2")