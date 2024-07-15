#!/bin/bash

#SBATCH --job-name=trainModel             # Process name
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
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <config_file> <optimizer_type> <batch_size> <learning_rate>"
    exit 1
fi

config_file=$1
optimizer_type=$2
batch_size=$3
learning_rate=$4

# Call the Python script with the provided arguments
python main.py $config_file $optimizer_type $batch_size $learning_rate
