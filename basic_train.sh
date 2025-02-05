#!/bin/bash

#SBATCH --job-name=run_train          # Job name
#SBATCH --account=cocoflops           # Account name
#SBATCH --partition=cocoflops         # Partition name
#SBATCH --time=12:00:00               # Run time (hh:mm:ss)
#SBATCH -c 12                          # Number of CPU cores
#SBATCH --gres=gpu:4                  # Number of GPUs
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --nodelist=cocoflops-hgx-1    # (Optional) Specific node; uncomment if needed
#SBATCH --mem=70G                     # Memory size
#SBATCH --output=./slurm_logs/%x_%j.log  # Unique log file per job: jobName_jobID.log

# source ~/.bashrc
# conda activate zero
# cd ~/TinyZero


export N_GPUS=4
export BASE_MODEL=/scr/kanishkg/ba/models/countdown_qwen2.5-3b_only_backtracking_sft/global_step_60
export DATA_DIR=/scr/kanishkg/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown_qwen2.5-3b_only_backtracking_ppo
export VLLM_ATTENTION_BACKEND=XFORMERS

sh ./scripts/train_tiny_zero_n4.sh
