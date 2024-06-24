#!/bin/bash

#SBATCH --job-name=homology               # Process name
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

# Call the Python script with the provided arguments
python homology_times.py

# Notify by email when the process is completed, not needed if SLURM mail is set
# mail -s "Proceso finalizado" pablolivares@correo.ugr.es <<< "El proceso ha finalizado"
