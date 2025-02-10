#!/bin/bash

process_model() {
    local model_name=$1
    local output_name="${model_name%_ppo}"
    
    local ckpt_base="/scr/akchak/rl_behaviors/checkpoints/${model_name}"
    local output_base="/scr/akchak/rl_behaviors/outputs/${output_name}"
    
    echo "Processing checkpoints from: ${ckpt_base}"
    echo "Outputting to: ${output_base}"
    
    mkdir -p "${output_base}"

    
    for step in $(seq 10 10 250); do
        local ckpt_path="${ckpt_base}/global_step_${step}"

        if [ ! -d "${ckpt_path}" ]; then
            echo "Checkpoint not found: ${ckpt_path}, skipping..."
            continue
        fi

        echo "Processing step ${step}..."
        python3 behavioral_evals/generate_completions.py \
            --ckpt "${ckpt_base}/global_step_${step}" \
            --dataset /scr/akchak/countdown/test.parquet \
            --output-path "${output_base}/"
            
        rm -rf "${ckpt_base}/global_step_${step}"
    done
    
    sbatch ./slurm_scripts/gpt_submit.sh "${output_name}"
}

echo "Processing only_backtracking_backward_ppo..."
process_model "only_backtracking_backward_ppo"
