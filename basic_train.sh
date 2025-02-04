export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=/scr/kanishkg/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b_ppo_n4
export VLLM_ATTENTION_BACKEND=XFORMERS

sh ./scripts/train_tiny_zero.sh
