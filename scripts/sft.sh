eval "$(conda shell.bash hook)"
conda activate zero

# Set environment variables
hf_cache_dir="/home/anikait.singh/.cache/"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539gmail.com
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

base_data_paths=(
  /home/anikait.singh/rl_behaviors/cot_datasets/rag_ai2_sft
)

# List of dataset conditions
conditions=(
  method
)

model_names=(
  Qwen/Qwen2.5-3B
)

epochs=(
  5
  10
)

lrs=(
  1e-5
  1e-6
)

# Shared training parameters
prompt_key="query"
response_key="completion"
micro_batch_size=16
train_batch_size=80
# max_length=2048
max_length=4096
default_hdfs_dir="/home/anikait.singh/rl_behaviors/hdfs"
default_local_dir="/home/anikait.singh/rl_behaviors/sft"
project_name="math-ragai2-sft-0223"
logger="['console','wandb']"
warmup_steps_ratio=0.05
weight_decay=1e-4

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi
# Iterate over each condition and launch a training job
for base_data_path in "${base_data_paths[@]}"; do
for model_name in "${model_names[@]}"; do
for condition in "${conditions[@]}"; do
for total_epochs in "${epochs[@]}"; do
for lr in "${lrs[@]}"; do
  if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
      exp_num=$((exp_num+1))
      continue
  fi
  
  train_file="${base_data_path}/${condition}/train.parquet"
  val_file="${base_data_path}/${condition}/test.parquet"

  model_name_short=$(echo $model_name | cut -d'/' -f2)
  data_path_short=$(echo $base_data_path | cut -d'/' -f6)
  experiment_name="sft-${model_name_short}-${data_path_short}-${condition}-${total_epochs}-${lr}-exp${exp_num}"
  # save_dir="${default_local_dir}/${model_name_short}/${condition}/${total_epochs}/${lr}/${exp_num}"
  save_dir="${default_local_dir}/${experiment_name}"
  mkdir -p $save_dir

  echo "Running training for condition: ${condition}"
  echo "Train file: ${train_file}"
  echo "Val file:   ${val_file}"
  echo "Experiment name: ${experiment_name}"
  echo ""

  command="torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.prompt_key="${prompt_key}" \
    data.response_key="${response_key}" \
    data.micro_batch_size="${micro_batch_size}" \
    data.train_batch_size="${train_batch_size}" \
    data.max_length="${max_length}" \
    data.truncation='right' \
    model.partial_pretrain="${model_name}" \
    trainer.default_hdfs_dir="${default_hdfs_dir}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.logger="${logger}" \
    optim.lr="${lr}" \
    optim.warmup_steps_ratio=$warmup_steps_ratio \
    optim.weight_decay=$weight_decay \
  "
  echo $command
  echo "--------------------------------------------------"


  if [ "$dry_run" = false ]; then
    eval $command
  fi

  exp_num=$((exp_num+1))
done
done
done
done
done