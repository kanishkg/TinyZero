import json
import argparse
import regex
from verl.utils.reward_score.countdown import compute_score
import os
import csv

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--input-file", '-i', type=str, required=True)
    argparser.add_argument("--output-file", '-o', type=str, required=True)

    args = argparser.parse_args()   
    data = json.load(open(args.input_file))

    scores = []
    for d in data:
        target_num = int(regex.findall(r'create an equation that equals (\d+)', d['query'])[0])
        numbers = [int(x.strip()) for x in regex.findall(r'\[(.*?)\]', d['query'])[0].split(',')]
        ground_truth = {'target': target_num, 'numbers': numbers}
        scores.append(compute_score(f"Assistant:\n{d['completion']}", ground_truth))
    
    condition = args.input_file.split('/')[-1].split('.')[0]
    print(f"Average score for {condition} condition: {sum(scores) / len(scores)}")

    file_exists = os.path.isfile(args.output_file)

    with open(args.output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['condition', 'average_score'])
        writer.writerow([condition, sum(scores) / len(scores)])


