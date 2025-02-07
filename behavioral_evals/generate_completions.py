import argparse
import os
import json
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from verl.utils.reward_score.countdown import compute_score
from tqdm import tqdm
import torch
import re

def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using vLLM on an HF parquet dataset and score them."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()
    ckpt_dir = args.ckpt
    dataset_path = args.dataset
    # Initialize the vLLM model from the checkpoint (using LLM instead of LLMEngine).
    llm = LLM(
        model=ckpt_dir,
        enable_prefix_caching=False,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )
    # Retrieve the tokenizer from the vLLM model.
    # tokenizer = llm.get_tokenizer()
    # Set up sampling parameters with stop tokens.
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=args.temperature,
    )
    # Load the parquet dataset.
    dataset = load_dataset("parquet", data_files=dataset_path)
    # List to hold all result records.
    # Iterate over the dataset examples.
    prompts = []
    gts = []
    for example in tqdm(dataset, desc="Generating responses"):
        # Extract the prompt.
        prompt = example["prompt"][0]["content"]
        # Extract the ground truth (for scoring) from the reward_model field.
        ground_truth = example["reward_model"]["ground_truth"]
        gts.append(ground_truth)
        # Extra information from the example (e.g., index and data source).
        extra_info = example.get("extra_info", {})
        index = extra_info.get("index", None)
        data_source = example.get("data_source", "")
        # Generate a response using vLLM.
        prompts.append(prompt)
    responses = llm.generate(prompts, sampling_params=sampling_params)

    results = []
    for ground_truth, prompt, response in zip(gts, prompts, responses):
        # Access the first generation from the response.
        generated_text = response.outputs[0].text.strip()
        generated_text = f"Assistant:\n{generated_text}"
        # Compute the reward score.
        score = compute_score(generated_text, ground_truth, format_score=0., score=1.)
        # Build the result record.
        result_record = {
            "index": index,
            "data_source": data_source,
            "prompt": prompt,
            "generated": generated_text,
            "score": score,
            "ground_truth": ground_truth,
        }
        results.append(result_record)
    # Save the results to a JSON file in the checkpoint directory.
    output_path = os.path.join(ckpt_dir, "responses.jsonl")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved responses and scores to {output_path}")

if __name__ == "__main__":
    main()