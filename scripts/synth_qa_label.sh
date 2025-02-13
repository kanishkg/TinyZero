CUDA_VISIBLE_DEVICES=0,1 python pretraining_analysis/relabel_olmo_pretrain_qa.py --user obiwan96 --start 0 --end 11616 --save_every 10000  & \
CUDA_VISIBLE_DEVICES=2,3 python pretraining_analysis/relabel_olmo_pretrain_qa.py --user obiwan96 --start 11616 --end 23232 --save_every 10000  & \
CUDA_VISIBLE_DEVICES=4,5 python pretraining_analysis/relabel_olmo_pretrain_qa.py --user obiwan96 --start 23232 --end 34848 --save_every 10000  & \
CUDA_VISIBLE_DEVICES=6,7 python pretraining_analysis/relabel_olmo_pretrain_qa.py --user obiwan96 --start 34848 --end 46467  --save_every 10000
