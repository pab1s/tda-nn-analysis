#!/bin/bash

#SBATCH --job-name=gradCAM                # Process name
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

# Check if correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_file> <model_path>"
    exit 1
fi

config_file=$1
model_path=$2

# Call the Python script with the provided arguments
python grad-cam.py $config_file $model_path
