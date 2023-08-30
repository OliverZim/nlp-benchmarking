#!/bin/bash
# add the arguments you want to test here
# example:
declare -A arguments
arguments['0']="--wandb_run_name=run0 --batch_size_per_device=8 --workers=1 --precision=32 --compile=false --force_deterministic=false"
arguments['1']="--wandb_run_name=run1 --batch_size_per_device=4 --workers=1 --precision=32 --compile=false --force_deterministic=false"

