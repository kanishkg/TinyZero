import argparse
import subprocess

# conda activate myenv; python /home/anikait.singh/TinyZero/pretraining_analysis/scripts.py --node 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=400000,
                        help='Starting shard index (inclusive)')
    parser.add_argument('--end', type=int, default=600000,
                        help='Ending shard index (exclusive)')
    parser.add_argument('--num_nodes', type=int, default=2,
                        help='Total number of nodes to use')
    parser.add_argument('--node', type=int, default=0,
                        help='Index of the current node (0-indexed)')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs available on this node')
    parser.add_argument('--tp_size', type=int, default=2,
                        help='Tensor parallelism size (number of GPUs per process)')
    parser.add_argument('--save_every', type=int, default=5000,
                        help='Save every N examples')
    parser.add_argument('--user', type=str, default='Asap7772',
                        help='User to push the dataset to')
    args = parser.parse_args()

    # Basic sanity checks.
    assert args.start < args.end, "Start must be less than end."
    assert args.node < args.num_nodes, "Node index must be less than total num_nodes."
    assert args.num_gpus % args.tp_size == 0, \
        "Number of GPUs must be divisible by tp_size."

    num_proc = args.num_gpus // args.tp_size

    all_gpus = [str(i) for i in range(args.num_gpus)]
    gpus = []
    for j in range(0, args.num_gpus, args.tp_size):
        gpus.append(','.join(all_gpus[j:j+args.tp_size]))

    total_shards = args.end - args.start
    base_shards_per_node = total_shards // args.num_nodes
    remainder_shards = total_shards % args.num_nodes

    if args.node < remainder_shards:
        node_start = args.start + args.node * (base_shards_per_node + 1)
        node_end = node_start + base_shards_per_node + 1
    else:
        node_start = args.start + remainder_shards * (base_shards_per_node + 1) \
                     + (args.node - remainder_shards) * base_shards_per_node
        node_end = node_start + base_shards_per_node

    node_shards = node_end - node_start

    base_shards_per_proc = node_shards // num_proc
    remainder_proc = node_shards % num_proc

    for i in range(num_proc):
        if i < remainder_proc:
            proc_start = node_start + i * (base_shards_per_proc + 1)
            proc_end = proc_start + base_shards_per_proc + 1
        else:
            proc_start = node_start + remainder_proc * (base_shards_per_proc + 1) \
                         + (i - remainder_proc) * base_shards_per_proc
            proc_end = proc_start + base_shards_per_proc

        if i == num_proc - 1:
            proc_end = node_end

        curr_gpu = gpus[i]
        env_prefix = f'CUDA_VISIBLE_DEVICES={curr_gpu} '
        command = (
            f'python /home/anikait.singh/TinyZero/pretraining_analysis/relabel_olmo_pretrain_qa.py '
            f'--start {proc_start} --end {proc_end} --user {args.user} --save_every {args.save_every}'
        )
        full_command = env_prefix + command

        subprocess.Popen(full_command, shell=True)
        print(f'Running command: {full_command}')
        print(f'Processing shards from {proc_start} to {proc_end}')
        print()

    print('All processes started.')

if __name__ == '__main__':
    main()