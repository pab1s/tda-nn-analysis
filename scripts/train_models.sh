#!/bin/bash

#SBATCH --job-name=train_EfficientNetB0    # Process name
#SBATCH --partition=dios                  # Queue for execution
#SBATCH -w atenea                         # Node to execute the job
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

# Notify by email when the process is completed, not needed if SLURM mail is set
# mail -s "Proceso finalizado" pablolivares@correo.ugr.es <<< "El proceso ha finalizado"
