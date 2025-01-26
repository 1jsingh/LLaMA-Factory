#!/bin/bash -l
#SBATCH --account=genai_interns
#SBATCH --qos=genai_interns
#SBATCH --partition=q2
#SBATCH --job-name=qwen25coder_full_sft
#SBATCH --output=slurm_outputs/qwen_finetune_%A_%j.out
#SBATCH --error=slurm_outputs/qwen_finetune_%A_%j.err
#SBATCH --time=23:59:59
#SBATCH --mem=384G
#SBATCH --nodes=1              # Number of nodes
#SBATCH --gpus-per-node=8        # GPUs per node
#SBATCH --cpus-per-task=64       # CPUs per task

# Exit on errors or undefined variables, and print commands
set -euo pipefail
set -x

# (1) Activate your conda environment
source /data/home/jaskirats/miniconda3/etc/profile.d/conda.sh
conda activate r2e

# Debug prints to verify environment
echo "Conda executable: $(which conda)"
conda info
echo "Python executable: $(which python)"
python --version

# (2) Get node info
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"

# (3) Get IP address of the master (head) node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

# (4) Export environment variables for PyTorch’s distributed run
export MASTER_ADDR="$head_node_ip"
export MASTER_PORT=29500
export NNODES="$SLURM_JOB_NUM_NODES"
export NODE_RANK="$SLURM_NODEID"

# (5) Optional: debug-level info for NCCL (helps diagnose multi-node issues)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
# If you suspect Infiniband issues, you can also try:
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0

# (6) (Optional) If you use Weights & Biases
export WANDB_PROJECT='r2e-edits'

# (7) Figure out where llamafactory’s launcher script is
LAUNCHER_PATH=$(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")
echo "Launcher path: $LAUNCHER_PATH"

# (8) Launch training
srun torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$LAUNCHER_PATH" examples/train_full/qwen25coder_full_sft.yaml