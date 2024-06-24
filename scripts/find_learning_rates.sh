#!/bin/bash

#SBATCH --job-name=train_EfficientNetB0    # Process name
#SBATCH --partition=dios                  # Queue for execution
#SBATCH --gres=gpu:1                      # Number of GPUs to use
#SBATCH --mail-type=END,FAIL              # Notifications for job done & fail
#SBATCH --mail-user=pablolivares@correo.ugr.es  # Where to send notification

# Load necessary paths
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

# Setup Conda environment
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/polivares/tda-nn/tda-nn-separability
export TFHUB_CACHE_DIR=.

# Check if correct number of arguments is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

config_file=$1

python find_learning_rates.py $config_file

# mail -s "Proceso finalizado" pablolivares@correo.ugr.es <<<"El proceso ha finalizado"
