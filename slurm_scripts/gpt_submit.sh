#!/bin/bash
#SBATCH --job-name=process_batch
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --time=24:15:00
#SBATCH -c 1
#SBATCH --nodes=1
#SBATCH --nodelist=cocoflops2
#SBATCH --mem=8G
#SBATCH --output=./slurm_logs/%x_%j.log

if [ $# -lt 2 ]; then
    echo "Error: Please provide an identifier as an argument"
    echo "Usage: sbatch $0 <identifier> <task-type>" 
    exit 1
fi

IDENTIFIER="$1"
TASK_TYPE="$2"

INPUT_DIR="/scr/akchak/rl_behaviors/outputs/${IDENTIFIER}/"
OUTPUT_DIR="/scr/akchak/rl_behaviors/outputs/${IDENTIFIER}/"

cd ~/TinyZero

python behavioral_evals/gpt_api_eval.py \
    --input "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --api-key api_key \
    --num-samples 200 \
    --task-type "${TASK_TYPE}"