#!/usr/bin/env python3
import os
import json
import csv
import sys
import argparse

def load_all_results(root_dir):
    """
    Walks the immediate subdirectories of root_dir.
    For each subdirectory, if an 'all_results.json' file exists, it is loaded via json.load.
    Returns a dictionary mapping the subdirectory name (condition) to the loaded JSON data.
    """
    results = {}
    for entry in os.listdir(root_dir):
        subdir = os.path.join(root_dir, entry)
        if os.path.isdir(subdir):
            json_path = os.path.join(subdir, "all_results.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    results[entry] = data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {json_path}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}", file=sys.stderr)
            else:
                print(f"Warning: {json_path} does not exist.", file=sys.stderr)
    return results

def process_data(results):
    """
    Given the results dictionary (mapping condition names to their respective data),
    this function iterates over each condition and its 'steps_processed' list.
    Only steps that are strings are processed.
    For each such step, the function extracts the corresponding metrics from the
    'results_by_step' subdictionary and returns a list of flattened rows.
    """
    rows = []
    for condition, cond_data in results.items():
        steps = cond_data.get("steps_processed", [])
        results_by_step = cond_data.get("results_by_step", {})
        for step in steps:
            if not isinstance(step, str):
                continue  # Skip steps that are not strings.
            if step in results_by_step:
                metrics = results_by_step[step]
                # Build a row that includes the condition, step, and all metric values.
                row = {"condition": condition, "step": step}
                row.update(metrics)
                rows.append(row)
            else:
                print(f"Warning: step '{step}' not found in results_by_step for condition '{condition}'.",
                      file=sys.stderr)
    return rows

def main():
    parser = argparse.ArgumentParser(
        description="Collate all_results.json files from a folder and flatten the data into a CSV "
                    "dataset (only including steps that are strings)."
    )
    parser.add_argument("input",
                        help="Input folder containing condition subdirectories (e.g., 'outputs')")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV file for the flattened dataset")
    args = parser.parse_args()

    # Verify that the input is a directory.
    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Collate the results from each subdirectory.
    results = load_all_results(args.input)
    if not results:
        print("No results were loaded. Check that your subdirectories contain all_results.json files.",
              file=sys.stderr)
        sys.exit(1)

    # Process the collated data to flatten it.
    rows = process_data(results)
    breakpoint()
    if not rows:
        print("No rows were produced. Check that your input data has steps_processed as strings.",
              file=sys.stderr)
        sys.exit(1)

    # Determine the fieldnames from the first row.
    fieldnames = list(rows[0].keys())

    # Write the flattened data to the specified CSV file.
    try:
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote {len(rows)} rows to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
