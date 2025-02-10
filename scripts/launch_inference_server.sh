eval "$(conda shell.bash hook)"
export OUTLINES_CACHE_DIR=/tmp/.outlines
conda activate sglang
export HF_TOKEN=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

# python -m sglang.launch_server --model-path Asap7772/sft-prm800k-llama31-8b-steptok --port 30000 --host 0.0.0.0 --dp-size=8
# python -m sglang.launch_server --model-path /home/anikait.singh/reasoning-value-verifiers/checkpoints/sft_checkpoints/sft-math-llamaft-1109/sft_lr1e-5_wd0.0_modelLlama-3.1-8B-Instruct_datasetmath-v2_schedulecosine_exp0_20241109_175903/checkpoint-85 --port 30000 --host 0.0.0.0 --dp-size=8
# python -m sglang.launch_server --model-path /home/anikait.singh/reasoning-value-verifiers/checkpoints/sft_checkpoints/sft-math-llamaft-1109/sft_lr1e-6_wd0.0_modelLlama-3.1-8B-Instruct_datasetmath-v2_schedulecosine_exp1_20241109_175906/checkpoint-85 --port 30000 --host 0.0.0.0 --dp-size=8
# python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B --port 30000 --host 0.0.0.0 --dp-size=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; 

python -m sglang.launch_server \
--model-path meta-llama/Llama-3.3-70B-Instruct \
--port 30000 \
--host "0.0.0.0" \
--dp-size=4 \
--tp-size=2