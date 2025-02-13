from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForCausalLM.from_pretrained(args.model_path)

model.push_to_hub(args.output_name)
tokenizer.push_to_hub(args.output_name)
print(f"Model and tokenizer uploaded to {args.output_name}")