#!/bin/bash

# Define the ranges for each parameter
learning_rates=(0.0001 0.001 0.01)
batch_sizes=(8 16 32 64)
num_epochs=(2 5 10)
optimizers=(Adam SGD)
loss_functions=(CrossEntropyLoss)

# Define the config file and output directory
config_file="config.yaml"
output_dir="experiments"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each combination of parameters
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for ne in "${num_epochs[@]}"; do
      for opt in "${optimizers[@]}"; do
        for lf in "${loss_functions[@]}"; do
        
          # Create a new config file with the modified parameters
          new_config_file="$output_dir/config_lr_${lr}_bs_${bs}_ne_${ne}_opt_${opt}_lf_${lf}.yaml"
          cp "$config_file" "$new_config_file"

          # Modify the parameters in the new config file
          sed -i "s/learning_rate:.*/learning_rate: $lr/" "$new_config_file"
          sed -i "s/batch_size:.*/batch_size: $bs/" "$new_config_file"
          sed -i "s/num_epochs:.*/num_epochs: $ne/" "$new_config_file"
          sed -i "s/optimizer:.*/optimizer:\n  type: \"$opt\"/" "$new_config_file"
          sed -i "s/loss_function:.*/loss_function:\n  type: \"$lf\"/" "$new_config_file"

          # Run the training process with the new config file
          python train.py --config "$new_config_file"
        done
      done
    done
  done
done