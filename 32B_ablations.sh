#!/bin/bash

# Set environment variables for single-node training
export FORCE_TORCHRUN=1
export NNODES=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NODE_RANK=0
export WANDB_PROJECT='r2e-edits'
export DISABLE_VERSION_CHECK=1

# Define the array of max_samples values
# samples_list=(100 200 400 800 1600)
samples_list=(1600 800 400 200 100)

# Retrieve the path to the launcher script (assumes llamafactory is installed)
# LAUNCHER_PATH=$(python -c "import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'launcher.py'))")

# Loop over each sample value and run the training sequentially
for SAMPLES in "${samples_list[@]}"; do
    echo "Launching training for max_samples=${SAMPLES}..."

    # Create a temporary config file for this SAMPLES value
    CONFIG_FILE="tmp_configs/32B_full_sft_${SAMPLES}.yaml"

    # Copy the original YAML to the temporary config
    cp examples/train_full/32B_full_sft.yaml "$CONFIG_FILE"

    # Update parameters in the temporary YAML config
    sed -i "s|^max_samples:.*|max_samples: ${SAMPLES}|" "$CONFIG_FILE"
    sed -i "s|^output_dir:.*|output_dir: saves/qwen_32B_ablations_maxsamples${SAMPLES}_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1|" "$CONFIG_FILE"
    sed -i "s|^run_name:.*|run_name: qwen_32B_ablations_maxsamples${SAMPLES}_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1|" "$CONFIG_FILE"

    # Run the training using torchrun on a single node with 4 GPUs
    llamafactory-cli train "$CONFIG_FILE"

done