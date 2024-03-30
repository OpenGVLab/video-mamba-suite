# Model Zoo

## Note

- For all the pretraining and finetuning, we adopt spaese/uniform sampling.
- `#Frame` $=$ `#input_frame` $\times$ `#crop` $\times$ `#clip`
- `#input_frame` means how many frames are input for model per inference
- `#crop` means spatial crops (e.g., 3 for left/right/center)
- `#clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

| Model                   | Pretrain-I | Pretrain-V | Finetuned | #Frame | Weight |
|-------------------------|------------|------------|-----------|--------|--------|
| ViViM-T                 | deit,IN1K  | K400   | -         | 16x3x4     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_tiny_k400_f16.pt)       |
| ViViM-S                 | deit,IN1K  | K400   | -         | 16x3x4     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_small_k400_f16.pt)       |


# Models and Usage

### Kinetics-400

| Method  | #Frame |  Top-1 Acc| Top-5 Acc | shell |
|---------|:----:|:---------:|:---------:|:---------:|
| ViViM-T (Ours) | 16 |   77.51   |   93.27   |  [script.sh](./exp/k400/vivim_tiny/run_f16x224.sh) | 
| ViViM-S (Ours) | 16 |   80.47   |   94.75   |  [script.sh](./exp/k400/vivim_small/run_f16x224.sh) | 