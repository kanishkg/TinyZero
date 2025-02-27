import argparse
import os
import re
import csv
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", help="Path to the results CSV")
    parser.add_argument("--condition", help="Condition to match, e.g. 'backtracking_subgoal'")
    parser.add_argument("--input-folder", help="Folder with completion_stepX.jsonl files")
    parser.add_argument("--tokenizer", choices=["qwen","llama"], required=True,
                        help="Tokenizer to use (similar to acc_resp.py)")
    args = parser.parse_args()

    if args.tokenizer == "qwen":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    elif args.tokenizer == "llama":
        from transformers import AutoTokenizer
        if '70b' in args.condition:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
        else:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    rows = []
    with open(args.csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pattern = re.compile(r"completions_step(\d+)")
    lengths_by_step = {}
    for filename in os.listdir(args.input_folder):
        match = pattern.search(filename)
        if match:
            step = int(match.group(1))
            path = os.path.join(args.input_folder, filename)

            with open(path, "r") as fin:
                data = json.load(fin)
            total_len = 0
            count = 0
            for d in data:
                response = d.get("generated", "")
                tokenized = tokenizer(response).input_ids
                total_len += len(tokenized)
                count += 1
            avg_len = total_len / count if count else 0
            lengths_by_step[step] = avg_len
    

    for row in rows:
        if row.get("condition") == args.condition:
            step_str = row.get("step", "")
            try:
                step_val = int(step_str)
                if step_val in lengths_by_step:
                    row["response_length"] = lengths_by_step[step_val]
            except ValueError:
                pass

    with open(args.csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
