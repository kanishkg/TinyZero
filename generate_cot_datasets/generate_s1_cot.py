import json

import datasets
import anthropic



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
- Consider working backwards from the goal state
- Break down complex goals into manageable sub-goals
- If stuck, return to a previous state and try a different approach

Format for writing solutions:
<think> thoughts here </think>
<answer> answer </answer>

Remember:
- Always verify your solution path
- Show clear reasoning for each step
- Backtrack when needed
- Document attempts and why certain paths weren't pursued
- Explain why certain sub-goals were chosen
"""
USER_PROMPT = "Solve the following problem: {question}"


client = anthropic.Anthropic()

s1ds = datasets.load_dataset("simplescaling/s1K")

questions = []
for sample in s1ds["train"]:
    question = sample["question"]
    questions.append(USER_PROMPT.format(question=question))

answers = []
for question in questions:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
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
        }
        ]
    )
    answers.append(message.content[0].text)

new_dataset = datasets.Dataset.from_dict({"question": questions, "answer": answers})
new_dataset.push_to_hub("obiwan96/s1-claude")