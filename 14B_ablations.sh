#!/bin/bash
#SBATCH --account=OD-227441
#SBATCH --job-name=qwen25coder_full_sft_ablations
#SBATCH --output=slurm_outputs_14B_ablations/qwen_finetune_%A_%a.out
#SBATCH --error=slurm_outputs_14B_ablations/qwen_finetune_%A_%a.err
#SBATCH --time=1:59:59
#SBATCH --mem=384G
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=h24gpu
#SBATCH --array=0-5

# 1) Load any necessary modules
module load cuda/12.2.2

# 2) Activate the Conda environment
source /scratch3/zha439/miniconda3/bin/activate
conda activate r2e

# 3) Prepare environment variables for distributed training
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=("${nodes[@]}")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"

export FORCE_TORCHRUN=1
export NNODES="$SLURM_JOB_NUM_NODES"
export MASTER_ADDR="$head_node_ip"
export MASTER_PORT=29500
export NODE_RANK="$SLURM_NODEID"
export WANDB_PROJECT='r2e-edits'
export DISABLE_VERSION_CHECK=1

# 4) Define the array of max_samples
samples_list=(100 200 400 800 1600 3200)

# Retrieve the index (0 to 5) and pick the correct value
SAMPLES="${samples_list[$SLURM_ARRAY_TASK_ID]}"

# 5) Define the path to the launcher script
LAUNCHER_PATH=$(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")

# 6) Create a temporary config file for this particular SAMPLES value
CONFIG_FILE="tmp_configs/14B_full_sft_${SAMPLES}.yaml"

# Copy the original YAML to the temp config
cp examples/train_full/14B_full_sft.yaml "$CONFIG_FILE"

# Update max_samples in the YAML
sed -i "s|^max_samples:.*|max_samples: ${SAMPLES}|" "$CONFIG_FILE"

# Update output_dir in the YAML
sed -i "s|^output_dir:.*|output_dir: saves/14B_ablations_maxsamples${SAMPLES}_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1|" "$CONFIG_FILE"

# Update run_name in the YAML (if you have a line like `run_name:` in there)
sed -i "s|^run_name:.*|run_name: 14B_ablations_maxsamples${SAMPLES}_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1|" "$CONFIG_FILE"

echo "Launching training for max_samples=${SAMPLES}..."

# 7) Run distributed training via srun + torchrun
srun torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$RANDOM" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$LAUNCHER_PATH" "$CONFIG_FILE"