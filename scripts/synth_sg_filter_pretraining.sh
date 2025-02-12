CUDA_VISIBLE_DEVICES=0,1 python pretraining_analysis/relabel_olmo_pretrain_offline.py --user obiwan96 --start 0 --end 194250 --save_every 10000 --only_subgoal & \
CUDA_VISIBLE_DEVICES=2,3 python pretraining_analysis/relabel_olmo_pretrain_offline.py --user obiwan96 --start 194250 --end 388500 --save_every 10000 --only_subgoal & \
CUDA_VISIBLE_DEVICES=4,5 python pretraining_analysis/relabel_olmo_pretrain_offline.py --user obiwan96 --start 388500 --end 582750 --save_every 10000 --only_subgoal & \
CUDA_VISIBLE_DEVICES=6,7 python pretraining_analysis/relabel_olmo_pretrain_offline.py --user obiwan96 --start 582750 --end 777000  --save_every 10000 --only_subgoal
