#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi

# add your benchmarks here:
# examples:
# python train.py --val_before_training=False --training_goal=60000 -d ./data/sw --wandb_run_name="precision16" --precision=16-mixed
# python train.py --val_before_training=False --training_goal=60000 -d ./data/sw --wandb_run_name="precision16mixed" --precision=bf16-mixed
# python train.py --val_before_training=False --training_goal=60000 -d ./data/sw --wandb_run_name="precision32" --precision=32  #default

