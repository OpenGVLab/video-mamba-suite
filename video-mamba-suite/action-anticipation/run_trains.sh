#!/bin/bash

i_values=(1 2)
j_values=(10 15 20 25 28)

for i in "${i_values[@]}"; do
  for j in "${j_values[@]}"; do
    command="srun -p Gvlab-S1 -N 1 --gres=gpu:1 --cpus-per-task 16 --quotatype=auto --async python tools/train_net.py --config_file ./configs/EK100/TESTRA/ablation/mamba_${i}_short_${j}.yaml"
    echo "Running: $command"
    eval $command
  done
done