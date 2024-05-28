#!/bin/bash

#SBATCH --job-name LR_EfficientNetB0    # Nombre del proceso

#SBATCH --partition dios                # Cola para ejecutar

#SBATCH --gres=gpu:1                    # Numero de gpus a usar

        

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/polivares/tda-nn-separability

export TFHUB_CACHE_DIR=.

python find_learning_rates.py          

mail -s "Proceso finalizado" pablolivares@correo.ugr.es <<< "El proceso ha finalizado"