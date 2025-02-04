#!/usr/bin/env bash

python3 examples/data_preprocess/countdown.py \
  --cot_dir /scr/akchak/rl_behaviors/cot_datasets/raw_data/backtracking_verification.jsonl \
  --local_dir /scr/akchak/rl_behaviors/cot_datasets/processed_data/backtracking_verification \
  --train_size 1000 \
  --test_size 200

python3 examples/data_preprocess/countdown.py \
  --cot_dir /scr/akchak/rl_behaviors/cot_datasets/raw_data/backtracking_backward.jsonl \
  --local_dir /scr/akchak/rl_behaviors/cot_datasets/processed_data/backtracking_backward \
  --train_size 1000 \
  --test_size 200

python3 examples/data_preprocess/countdown.py \
  --cot_dir /scr/akchak/rl_behaviors/cot_datasets/raw_data/backtracking_subgoal.jsonl \
  --local_dir /scr/akchak/rl_behaviors/cot_datasets/processed_data/backtracking_subgoal \
  --train_size 1000 \
  --test_size 200

python3 examples/data_preprocess/countdown.py \
  --cot_dir /scr/akchak/rl_behaviors/cot_datasets/raw_data/only_backtracking.jsonl \
  --local_dir /scr/akchak/rl_behaviors/cot_datasets/processed_data/only_backtracking \
  --train_size 1000 \
  --test_size 200

python3 examples/data_preprocess/countdown.py \
  --cot_dir /scr/akchak/rl_behaviors/cot_datasets/raw_data/all_strategies.jsonl \
  --local_dir /scr/akchak/rl_behaviors/cot_datasets/processed_data/all_strategies \
  --train_size 1000 \
  --test_size 200

