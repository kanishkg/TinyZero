import sys
import json

sys.path.append('./OpenRLHF')
from tasks.countdown import CountDown

def main():
    checker = CountDown.verify_answer
    jsonl_file = sys.argv[1]
    total_accuracy = 0
    total_count = 0
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            completion = data["completion"]
            accuracy = checker(query, completion)
            total_accuracy += accuracy
            total_count += 1
            print(f"Accuracy: {accuracy}")
    print(f"Total accuracy: {total_accuracy / total_count}: {int(total_accuracy)}/{total_count}")

if __name__ == "__main__":
    main()