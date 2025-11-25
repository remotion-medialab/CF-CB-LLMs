#!/bin/bash
#SBATCH -J train_cb_llm             # Job name
#SBATCH -o logs/train_cb_llm.out          # Standard output and error log
#SBATCH -e logs/train_cb_llm.err          # Standard error log
#SBATCH -p mit_normal_gpu      # Partition (queue) name # mit_normal_gpu sched_mit_hill
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 4                   # Number of CPU cores
#SBATCH --gres=gpu:2                 # if you need GPUs
#SBATCH -t 3:00:00            # Runtime in HH:MM:SS (adjust as needed)
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pdong@mit.edu @ Change to your email
# Load required modules (if any)
# module load python  # Adjust based on your Python version

# Activate your Conda environment
# source ~/.zshrc
conda init zsh

# source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh # Adjust path if Conda is installed elsewhere
conda activate UROPEnv


# Run your script
SWEEP_ID="peterdong1951-massachusetts-institute-of-technology/CB-LLM-Classification-Test/o5cvvyrb" # Change to your sweep project

wandb agent "$SWEEP_ID"
