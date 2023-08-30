#!/bin/bash -eux
#SBATCH --container-image=ADD FULL PATH TO ENROOT IMAGE HERE
#SBATCH --container-workdir=ADD FULL PATH TO CURRENT WORKING DIRECTORY HERE
#SBATCH --container-mounts=./data:/data
#SBATCH --container-env="WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY"
#SBATCH --partition=FILL IN ARGUMENT
#SBATCH --cpus-per-task=FILL IN ARGUMENT # -c
#SBATCH --gpus=FILL IN ARGUMENT # -G
#SBATCH --mem=FILL IN ARGUMENT # -m
#SBATCH --time=FILL IN ARGUMENT
#SBATCH --array=SPECIFY RANGE OF BENCHMARKS AND HOW MANY TO RUN IN PARALLEL (e.g. --array=0-9%2 will run 10 Jobs but only 2 at most at the same time)

# Run python script
source ./config.sh
/opt/conda/envs/research/bin/python3  ./train.py --training_goal=SET_TRAINING_GOAL -d ./data/SET_DATA_DIR ${arguments[${SLURM_ARRAY_TASK_ID}]}

