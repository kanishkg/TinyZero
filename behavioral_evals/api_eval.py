import json
from google import genai
from argparse import ArgumentParser
import os
import re
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
sys.path.append('./OpenRLHF')
from tasks.countdown import CountDown

GEMINI_API_KEY = "AIzaSyBt1BuUCQvwYMJqRu7Pj84xKpi8bnEt5dM"

class RateLimitedEvaluator:
    def __init__(self, api_key: str, rate_limit: int = 500, target_utilization: float = 0.9):
        self.client = genai.Client(api_key=api_key)
        self.rate_limit = rate_limit  # requests per minute
        self.target_utilization = target_utilization
        self.target_requests_per_minute = rate_limit * target_utilization
        self.request_times = []
        self.last_request_time = None
        self.target_interval = 60 / self.target_requests_per_minute
        self.checker = CountDown.verify_answer

    async def _maintain_request_rate(self):
        """Maintain a steady request rate at target utilization"""
        now = datetime.now()
        
        if self.last_request_time is not None:
            # Calculate time since last request
            elapsed = (now - self.last_request_time).total_seconds()
            
            # If we're going faster than our target rate, wait
            if elapsed < self.target_interval:
                await asyncio.sleep(self.target_interval - elapsed)
        
        # Update last request time after any necessary wait
        self.last_request_time = datetime.now()
        
        # Maintain rolling window of request times
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        self.request_times.append(now)
        
        # If we're somehow over rate limit, implement emergency brake
        if len(self.request_times) >= self.rate_limit:
            wait_time = (self.request_times[0] - minute_ago).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_times = self.request_times[1:]

    async def _make_rate_limited_request(self, prompt: str) -> str:
        """Make a rate-limited request to the Gemini API"""
        await self._maintain_request_rate()
        
        # Make API request
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
        
        return response.text

    async def _verify_with_gemini(self, query: str, completion: str) -> Dict[str, Any]:
        """Verify multiple behaviors using Gemini model with rate limiting"""
        # Extract numbers and target from query
        target = int(re.search(r'results in (\d+)', query).group(1))
        numbers = [int(x.strip()) for x in re.search(r'numbers ([\d, ]+)', query).group(1).split(',')]
        
        # 1. Check for answer verification steps
        verification_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}. 

Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' or 'Since 25 is not equal to 22'. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number. 

If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>."""

        verification_response = await self._make_rate_limited_request(verification_prompt)
        verification_count = int(re.search(r'<count>(\d+)</count>', verification_response).group(1))

        # 2. Check for backtracking behavior
        backtracking_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. An example of backtracking is: "Let me try again" or "we need to try a different sequence". We want to mark instances where the chain-of-reasoning is abandoned and the model backtracks to a previous computation. 

Count the number of distinct backtracking instances and provide the count between the tags <count> </count>."""

        backtracking_response = await self._make_rate_limited_request(backtracking_prompt)
        backtracking_count = int(re.search(r'<count>(\d+)</count>', backtracking_response).group(1))

        # 3. Check for subgoal setting
        subgoal_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll try to get close to {target//2}, then...".

Count the number of distinct subgoals set and provide the count between the tags <count> </count>."""

        subgoal_response = await self._make_rate_limited_request(subgoal_prompt)
        subgoal_count = int(re.search(r'<count>(\d+)</count>', subgoal_response).group(1))

        # 4. Check for backward-chaining behavior
        backward_chaining_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

    Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".

    Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>."""
        backward_response = await self._make_rate_limited_request(backward_chaining_prompt)
        backward_count = int(re.search(r'<count>(\d+)</count>', backward_response).group(1))
        
        # Calculate accuracy
        accuracy = self.checker(query, completion)
        
        return {
            'verification_count': verification_count,
            'backtracking_count': backtracking_count,
            'subgoal_count': subgoal_count,
            'backward_count': backward_count,
            'accuracy': accuracy
        }

    async def process_completions(self, completions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process all completions with concurrent rate limiting"""
        # Calculate target concurrent tasks based on rate limit
        # We want enough concurrent tasks to maintain our target rate
        # Each completion needs 4 requests, so we adjust accordingly
        target_concurrency = int(self.target_requests_per_minute / 60 * 4)  # 4 seconds worth of requests
        
        # Calculate and display estimated completion time
        total_requests = len(completions) * 4  # 4 prompts per completion
        estimated_time = (total_requests * self.target_interval) / (60 * target_concurrency)  # minutes
        print(f"Estimated completion time: {estimated_time:.2f} minutes with {target_concurrency} concurrent tasks")
        
        results = []
        semaphore = asyncio.Semaphore(target_concurrency)
        
        async def process_with_semaphore(item):
            async with semaphore:
                try:
                    result = await self._verify_with_gemini(item['query'], item['completion'])
                    return {
                        'query': item['query'],
                        'completion': item['completion'],
                        **result
                    }
                except Exception as e:
                    print(f"Error processing item: {e}")
                    return None
        
        # Create tasks for all completions
        tasks = [process_with_semaphore(item) for item in completions]
        
        # Process in batches with progress bar
        with tqdm(total=len(completions), desc="Processing completions") as pbar:
            for batch_start in range(0, len(tasks), 100):
                batch = tasks[batch_start:batch_start + 100]
                batch_results = await asyncio.gather(*batch)
                
                # Filter out None results from errors
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)
                pbar.update(len(batch))
                
                # Print current utilization stats
                current_rate = len(self.request_times)
                utilization = current_rate / self.rate_limit
                print(f"\nCurrent utilization: {utilization:.2%} ({current_rate} requests/minute)")
        
        return results

def get_completion_files(completions_dir: str) -> List[tuple[int, str]]:
    """Get sorted list of completion files with their step numbers"""
    files = []
    for filename in os.listdir(completions_dir):
        if filename.startswith('completions_step') and filename.endswith('.jsonl'):
            match = re.search(r'completions_step(\d+)\.jsonl', filename)
            if match:
                step = int(match.group(1))
                files.append((step, os.path.join(completions_dir, filename)))
    return sorted(files)  # Sort by step number

async def main():
    parser = ArgumentParser(description="Evaluate completions using Gemini API")
    parser.add_argument(
        "--completions-dir",
        type=str,
        required=True,
        help="Directory containing completion JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=GEMINI_API_KEY,
        help="Gemini API key"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all completion files
    if not args.completions_dir.endswith('.jsonl'):
        completion_files = get_completion_files(args.completions_dir)
    else:
        completion_files = [(0, args.completions_dir)]
    print(f"Found {len(completion_files)} completion files")
    
    # Initialize evaluator
    evaluator = RateLimitedEvaluator(args.api_key)
    
    # Process each file
    all_results = {}
    for step, filepath in completion_files:
        print(f"\nProcessing step {step} from {os.path.basename(filepath)}")
        
        # Load completions
        completions = []
        with open(filepath, 'r') as f:
            for line in f:
                completions.append(json.loads(line))
        
        # Process completions
        print(f"Processing {len(completions)} completions...")
        results = await evaluator.process_completions(completions)
        
        # Calculate metrics for this step
        total_verifications = sum(r['verification_count'] for r in results)
        total_backtracking = sum(r['backtracking_count'] for r in results)
        total_subgoals = sum(r['subgoal_count'] for r in results)
        
        avg_verifications = total_verifications / len(results)
        avg_backtracking = total_backtracking / len(results)
        avg_subgoals = total_subgoals / len(results)
        accuracy = sum(r['accuracy'] for r in results) / len(results)
        
        # Store results for this step
        step_results = {
            'results': results,
            'metrics': {
                'total_verifications': total_verifications,
                'total_backtracking': total_backtracking,
                'total_subgoals': total_subgoals,
                'avg_verifications': avg_verifications,
                'avg_backtracking': avg_backtracking,
                'avg_subgoals': avg_subgoals,
                'accuracy': accuracy
            }
        }
        
        # Save individual step results
        output_path = os.path.join(
            args.output_dir,
            f"evaluation_step{step}.json"
        )
        
        print(f"Storing results at: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(step_results, f, indent=2)
            
        # Store in aggregate results
        all_results[step] = {
            'avg_verifications': avg_verifications,
            'avg_backtracking': avg_backtracking,
            'avg_subgoals': avg_subgoals,
            'total_verifications': total_verifications,
            'total_backtracking': total_backtracking,
            'total_subgoals': total_subgoals,
            'accuracy': accuracy
        }
        
        print(f"Step {step} complete:")
        print(f"Average verifications: {avg_verifications:.4f}")
        print(f"Average backtracking: {avg_backtracking:.4f}")
        print(f"Average subgoals: {avg_subgoals:.4f}")
        print(f"Total verifications: {total_verifications}")
        print(f"Total backtracking: {total_backtracking}")
        print(f"Total subgoals: {total_subgoals}")
        print(f"Accuracy: {accuracy:.4f}")
    
    # Save aggregate results
    aggregate_path = os.path.join(args.output_dir, "all_results.json")
    with open(aggregate_path, 'w') as f:
        json.dump({
            'results_by_step': all_results,
            'steps_processed': sorted(all_results.keys())
        }, f, indent=2)
    
    print("\nAll processing complete!")
    print(f"Full results saved to: {aggregate_path}")

if __name__ == "__main__":
    asyncio.run(main())