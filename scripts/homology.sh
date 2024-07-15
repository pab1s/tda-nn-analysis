#!/bin/bash

#SBATCH --job-name=homology               # Process name
#SBATCH --partition=dios                  # Queue for execution
#SBATCH -w dionisio                       # Node to execute the job
#SBATCH --gres=gpu:1                      # Number of GPUs to use
#SBATCH --mail-type=END,FAIL              # Notifications for job done & fail
#SBATCH --mail-user=user@mail.com         # Where to send notification

# Load necessary paths
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=$(dirname $(dirname "$0"))

# Setup Conda environment
eval "$(conda shell.bash hook)"
conda activate tda-nn-analysis
export TFHUB_CACHE_DIR=.

# Ensure that the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_name> <model_name>"
    exit 1
fi

# Read the command-line arguments
CONFIG_NAME=$1
MODEL_NAME=$2

# Call the Python script with the provided arguments
python homology_times.py $CONFIG_NAME $MODEL_NAME
