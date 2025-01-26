#!/bin/bash -l
#SBATCH --account=genai_interns
#SBATCH --qos=genai_interns
#SBATCH --partition=q2
#SBATCH --job-name=qwen25coder_full_sft
#SBATCH --output=slurm_outputs/qwen_finetune_%A_%j.out
#SBATCH --error=slurm_outputs/qwen_finetune_%A_%j.err
#SBATCH --time=23:59:59
#SBATCH --mem=384G
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64

set -euo pipefail
set -x

# Option 1: if conda is definitely set up in .bashrc
source /data/home/jaskirats/miniconda3/etc/profile.d/conda.sh 
conda activate r2e

echo "which conda: $(which conda)"
conda info
echo "which python: $(which python)"
python --version
# && cd /data/home/jaskirats/project/r2e/r2e-edits-internal

# Retrieve the list of allocated nodes
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=(${nodes[@]})
head_node=${nodes_array[0]}

# Retrieve the master node's IP address using srun
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"

# Export environment variables required by LlamaFactory
export FORCE_TORCHRUN=1
export NNODES=$SLURM_JOB_NUM_NODES
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export WANDB_PROJECT='r2e-edits'

echo "Environment Variables:"
echo "FORCE_TORCHRUN=$FORCE_TORCHRUN"
echo "NNODES=$NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "WANDB_PROJECT=$WANDB_PROJECT"

# Determine the path to LlamaFactory's internal launcher script
LAUNCHER_PATH=$(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")

echo "Launcher path: $LAUNCHER_PATH"

# Launch training with torchrun
srun torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $LAUNCHER_PATH examples/train_full/qwen25coder_full_sft.yaml
    # $LAUNCHER_PATH examples/train_full/llama3_full_sft.yaml