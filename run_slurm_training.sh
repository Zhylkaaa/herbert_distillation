#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=distil-herbert
#SBATCH -p nvidia
#SBATCH --cpus-per-task=64
#SBATCH --mem=480GB
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4

#SBATCH --time=96:00:00

eval "$(conda shell.bash hook)"
conda activate distil

set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=29400
ip_head=$head_node_ip:$port
export ip_head

export WANDB_PROJECT=distil-herbert

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,WANDB_PROJECT="$WANDB_PROJECT" \
        accelerate launch --main_process_ip="$head_node_ip" \
                          --main_process_port="$port" \
                          --machine_rank="$i" \
                          --num_processes=8 \
                          --num_machines=2 \
                          --multi_gpu \
                          --mixed_precision=fp16 \
                          distilation_main.py \
                          --per_device_train_batch_size 64 \
                          --per_device_eval_batch_size 64 &
done

srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,WANDB_PROJECT="$WANDB_PROJECT" \
    accelerate launch --main_process_ip="$head_node_ip" \
                          --main_process_port="$port" \
                          --machine_rank="0" \
                          --num_processes=8 \
                          --num_machines=2 \
                          --multi_gpu \
                          --mixed_precision=fp16 \
                          distilation_main.py \
                          --per_device_train_batch_size 64 \
                          --per_device_eval_batch_size 64

