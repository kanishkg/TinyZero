model_paths=(
    /home/anikait.singh/rl_behaviors/sft/OLMo-7B-hf/backtrack/global_step_500
    /home/anikait.singh/rl_behaviors/sft/OLMo-7B-hf/baseline/global_step_500
    /home/anikait.singh/rl_behaviors/sft/OLMo-7B-hf/negative/global_step_500
    /home/anikait.singh/rl_behaviors/sft/OLMo-7B-hf/method/global_step_500

    /home/anikait.singh/rl_behaviors/sft/Llama-3.2-3B/backtrack/global_step_500
    /home/anikait.singh/rl_behaviors/sft/Llama-3.2-3B/baseline/global_step_500
    /home/anikait.singh/rl_behaviors/sft/Llama-3.2-3B/negative/global_step_500
    /home/anikait.singh/rl_behaviors/sft/Llama-3.2-3B/method/global_step_500
)

base_models=(
    allenai/OLMo-7B-hf
    allenai/OLMo-7B-hf
    allenai/OLMo-7B-hf
    allenai/OLMo-7B-hf

    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-3B
)

output_names=(
    Asap7772/olmo7b-pretdata-backtrack
    Asap7772/olmo7b-pretdata-baseline
    Asap7772/olmo7b-pretdata-negative
    Asap7772/olmo7b-pretdata-method

    Asap7772/llama3b-pretdata-backtrack
    Asap7772/llama3b-pretdata-baseline
    Asap7772/llama3b-pretdata-negative
    Asap7772/llama3b-pretdata-method
)

num_models=${#model_paths[@]}
num_base_models=${#base_models[@]}
num_output_names=${#output_names[@]}

if [ $num_models -ne $num_base_models ]; then
    echo "Number of models and base models should be the same"
    exit 1
fi

if [ $num_models -ne $num_output_names ]; then
    echo "Number of models and output names should be the same"
    exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_models-1))); do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    curr_model_path=${model_paths[i]}
    curr_base_model=${base_models[i]}
    curr_output_name=${output_names[i]}

    python /home/anikait.singh/TinyZero/pretraining_analysis/upload_to_hf.py \
        --model_path $curr_model_path \
        --base_model $curr_base_model \
        --output_name $curr_output_name
done