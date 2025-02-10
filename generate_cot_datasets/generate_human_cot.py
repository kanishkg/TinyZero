import argparse
import csv
import json
import ast
from typing import Dict, List

def parse_choices(choices_str: str) -> List[int]:
    """Parse the choices string into a list of numbers."""
    try:
        # Remove any whitespace and parse the string as a Python literal
        return ast.literal_eval(choices_str.strip())
    except:
        return []

def create_query(numbers: List[int], target: int) -> str:
    """Create the standardized query string."""
    return (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer.\n"
        f"User: Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can "
        "only be used once. Show your work in <think> </think> tags. And return "
        "the final answer in <answer> </answer> tags.\n"
        "Assistant: Let me solve this step by step."
    )

def create_completion(transcript: str, response: str) -> str:
    """Create the standardized completion string."""
    return (
        f"<think>\n{transcript.strip()}</think>\n"
        f"<answer>{response.strip()}</answer>"
    )

def process_row(row: Dict) -> Dict:
    """Process a single row of data into the desired format."""
    try:
        # Extract and validate required fields
        choices = parse_choices(row['choices'])
        target = float(row['target'])
        transcript = row['transcript'].strip()
        response = row['response'].strip()

        # Create the formatted output
        return {
            "query": create_query(choices, int(target)),
            "completion": create_completion(transcript, response)
        }
    except (KeyError, ValueError) as e:
        print(f"Error processing row: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert game CSV data to JSON format')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='output.json',
                       help='Output JSON file path (default: output.json)')
    
    args = parser.parse_args()

    try:
        # Read CSV file and collect all processed rows
        processed_data = []
        with open(args.input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                processed_row = process_row(row)
                if processed_row:
                    processed_data.append(processed_row)
        
        # Write all data as a single JSON array
        with open(args.output, 'w', encoding='utf-8') as json_file:
            json.dump(processed_data, json_file, indent=4)
        
        print(f"Successfully converted {args.input_file} to {args.output}")
    
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input_file}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()