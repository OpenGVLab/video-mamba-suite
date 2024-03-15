#!/bin/bash

i_values=(1 2)
j_values=(15 20 25 28)

for i in "${i_values[@]}"; do
  for j in "${j_values[@]}"; do
    command="srun -p Gvlab-S1 -N 1 --gres=gpu:1 --cpus-per-task 16 --quotatype=auto --async python tools/test_net.py --config_file ./configs/EK100/TESTRA/ablation/mamba_${i}_short_${j}.yaml MODEL.CHECKPOINT ./checkpoints/configs/EK100/TESTRA/ablation/mamba_${i}_short_${j}/epoch-17.pth MODEL.LSTR.INFERENCE_MODE batch"
    echo "Running: $command"
    eval $command
  done
done