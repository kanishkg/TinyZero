#!/bin/bash
process_model() {
    local model_name=$1
    local task_type=$2
    local output_name="${model_name%_ppo}"
    
    local ckpt_base="/scr/akchak/rl_behaviors/checkpoints/${model_name}"
    local output_base="/scr/akchak/rl_behaviors/outputs/${output_name}"
    
    echo "Processing checkpoints from: ${ckpt_base}"
    echo "Outputting to: ${output_base}"
    echo "Using task-type: ${task_type}"
    
    mkdir -p "${output_base}"
    
    for step in $(seq 0 10 250); do
        local ckpt_path="${ckpt_base}/global_step_${step}"
        if [ ! -d "${ckpt_path}" ]; then
            echo "Checkpoint not found: ${ckpt_path}, skipping..."
            continue
        fi
        echo "Processing step ${step}..."
        python3 behavioral_evals/generate_completions.py \
            --ckpt "${ckpt_base}/global_step_${step}" \
            --dataset /scr/akchak/countdown/test.parquet \
            --temperature 0.5 \
            --output-path "${output_base}/" \
            --task-type "${task_type}"
            
        # rm -rf "${ckpt_base}/global_step_${step}"
    done
    
    sbatch ./slurm_scripts/gpt_submit.sh "${output_name}" "${task_type}"
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task-type>"
    exit 1
fi

TASK_TYPE="math"

echo "Processing owm-method with task-type: ${TASK_TYPE}..."
process_model "owm-method" "${TASK_TYPE}"