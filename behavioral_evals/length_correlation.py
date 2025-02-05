import sys
import json
from scipy.stats import pearsonr, spearmanr

sys.path.append('./OpenRLHF')
from tasks.countdown import CountDown

def main():
    checker = CountDown.verify_answer
    jsonl_file = sys.argv[1]
    
    accuracies = []
    lengths = []

    total_accuracy = 0
    total_count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            completion = data["completion"]

            # Compute accuracy for the current sample
            accuracy = checker(query, completion)

            # Compute length of the completion in characters
            completion_len = len(completion)

            # Store values
            accuracies.append(accuracy)
            lengths.append(completion_len)

            # Keep track of overall accuracy
            total_accuracy += accuracy
            total_count += 1

            print(f"Accuracy: {accuracy:.2f}, Completion length: {completion_len}")

    # Final accuracy (average)
    avg_accuracy = total_accuracy / total_count if total_count > 0 else 0
    print(f"\nOverall Accuracy: {avg_accuracy:.3f} ({int(total_accuracy)}/{total_count})")

    # Compute Pearson and Spearman correlation
    pearson_corr, pearson_p = pearsonr(lengths, accuracies)
    spearman_corr, spearman_p = spearmanr(lengths, accuracies)

    print("\nCorrelation Results:")
    print(f"  Pearson correlation:  r = {pearson_corr:.4f}, p-value = {pearson_p:.4g}")
    print(f"  Spearman correlation: r = {spearman_corr:.4f}, p-value = {spearman_p:.4g}")

if __name__ == "__main__":
    main()
