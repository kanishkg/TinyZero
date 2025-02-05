eval "$(conda shell.bash hook)"
conda activate zero

# Set environment variables
hf_cache_dir="/raid0/.cache/"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539gmail.com
export WANDB__SERVICE_WAIT=300
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

models=(
    # qhduan/aquila-7b
    # Qwen/Qwen2.5-3B
    # allenai/OLMo-2-1124-7B
    # allenai/OLMo-7B-hf
    allenai/OLMo-1B-hf
    google/gemma-2b
)
num_models=${#models[@]}
names=(
    # countdown-aquila7b
    # countdown-qwen2.5-3b
    # countdown-olmo7b
    countdown-olmo1b
    countdown-gemma2b
)
num_names=${#names[@]}
data_dir="/raid0/data_countdown"

gpus=("0,1,2,3" "4,5,6,7")
num_gpus=${#gpus[@]}

if [ $num_models -ne $num_names ]; then
    echo "Number of models and names should be the same"
    exit 1
fi

if [ $num_models -ne $num_gpus ]; then
    echo "Number of models and gpus should be the same"
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

    export N_GPUS=4
    export BASE_MODEL=${models[$i]}
    export DATA_DIR=$data_dir
    export ROLLOUT_TP_SIZE=2
    export EXPERIMENT_NAME=${names[$i]}
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export CUDA_VISIBLE_DEVICES=${gpus[$i]}

    command="bash ./scripts/train_tiny_zero.sh"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    echo $command
    if [ $dry_run = true ]; then
        echo -e "Dry run. Skipping...\n\n"
    else
        eval $command
    fi
    
    exp_num=$((exp_num+1))
done
