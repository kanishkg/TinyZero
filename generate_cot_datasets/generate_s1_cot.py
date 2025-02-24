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
for sample in s1ds["train"]:
    question = sample["question"]
    questions.append(question)
    user_message = USER_PROMPT.format(question=question)
    requests.append(Request(
        messages=[MessageCreateParamsNonStreaming(role="user", content=user_message)],
        max_tokens=1600,
        temperature=0.7,
        system=PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }]
    ))

answers = []
message_batch = client.messages.create(requests=requests[:2])
for message in message_batch.messages:
    answers.append(message.content[0].text)

new_dataset = datasets.Dataset.from_dict({"question": questions, "answer": answers})
new_dataset.push_to_hub("obiwan96/s1-claude-v2")