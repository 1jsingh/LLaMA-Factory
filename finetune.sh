#!/bin/bash
#SBATCH --account=OD-227441
#SBATCH --job-name=llama_finetune
#SBATCH --output=slurm_outputs/llama_finetune_%A_%j.out
#SBATCH --error=slurm_outputs/llama_finetune_%A_%j.err
#SBATCH --time=1:59:59
#SBATCH --mem=200G
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# ----------------------------
# 1. Load Necessary Modules
# ----------------------------
module load cuda/12.2.2
source /scratch3/zha439/miniconda3/bin/activate
conda activate base

# ----------------------------
# 2. Retrieve Node Information
# ----------------------------

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=(${nodes[@]})
head_node=${nodes_array[0]}

# Retrieve the master node's IP address using srun
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"

# ----------------------------
# 3. Export Environment Variables
# ----------------------------

export FORCE_TORCHRUN=1
export NNODES=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID

echo "Environment Variables:"
echo "FORCE_TORCHRUN=$FORCE_TORCHRUN"
echo "NNODES=$NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"

# ----------------------------
# 4. Launch Training with srun
# ----------------------------

srun llamafactory-cli train examples/train_full/llama3_full_sft.yaml