# Model Zoo

| Model                   | Pretrain-I | Pretrain-V | Finetuned | #Frame | Weight |
|-------------------------|------------|------------|-----------|--------|--------|
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_ego4d4m_bs512_f4.pt)    |
| TimeSformer-B (Frozen)  | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_frozen_base_ego4d4m_bs512_f4.pt)       |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_ego4d4m_bs512_f4.pt)       |
| TimeMamba-B (Frozen)    | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_frozen_base_ego4d4m_bs512_f4.pt)      |
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | EK100-CLS | 4->16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_ego4d4m_bs512_f4_ft_ek100_cls_f16.pt)      |
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | EK100-MIR | 4->16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_ego4d4m_bs512_f4_ft_ek100_mir_f16.pt)      |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | EK100-CLS | 4->16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_ego4d4m_bs512_f4_ft_ek100_cls_f16.pt)      |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | EK100-MIR | 4->16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_ego4d4m_bs512_f4_ft_ek100_mir_f16.pt)      |
| ViT-T                   | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_tiny_ego4d4m_bs512_f16.pt)       |
| ViT-S                   | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_small_ego4d4m_bs512_f16.pt)       |
| ViViM-T                 | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_tiny_ego4d4m_bs512_f16.pt)       |
| ViViM-S                 | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_small_ego4d4m_bs512_f16.pt)       |


# Models and Usage



### 1. Zero-shot Multi-instance Retrieval on EK100


#### 1.1 Video Temporal Adapter


| Method | #Frame | V2T mAP |  T2V mAP |  Avg mAP | V2T nDCG | T2V nDCG | Avg nDCG | Train shell | Infer shell |
|-----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| TimeSformer (Vanilla) | 4 |   29.2   |   21.8   |   25.5   |   30.1   |   27.1   |   28.6   | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_timesformer.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_timesformer_infer_mir_f4.sh) |
| TimeSformer (Frozen)  | 4 |  29.8   |   22.2   |   26.0   |   30.6   |   27.5   |   29.0   | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_timesformer_frozenintime.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_timesformer_like_frozen_infer_mir_f4.sh) |
| TimeMamba (Vanilla, Ours)   | 4 |   30.3   |   22.1   |   26.2   |   30.9   |   27.5   |   29.2   | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_timemamba_like_timesformer.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_timemamba_infer_mir_f4.sh) |
| TimeMamba (Frozen, Ours)    | 4 | **30.7** | **22.8** | **26.8** | **31.3** | **27.8** | **29.5** | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_timemamba_like_frozen.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_timemamba_like_frozen_infer_mir_f4.sh) |

#### 1.2 Spatial-Temporal Modeling

| Method  | #Frame |  V2T mAP  |  T2V mAP  |  Avg mAP  |  V2T nDCG |  T2V nDCG |  Avg nDCG |  Train shell | Infer shell |
|---------|:----:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ViT-T   | 4  |   15.50   |   11.10   |   13.30   |   22.48   |   19.66   |   21.07   |
| ViT-B   | 4  |   25.08   |   18.49   |   21.79   |   27.80   |   24.87   |   26.34   | 
| ViT-T   | 16 |   20.47   |   15.29   |   17.88   |   25.74   |   22.89   |   24.31   | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_deit_tiny_gpu8_f16.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_vit_tiny_infer_mir_f16.sh) |
| ViT-S   | 16 |   23.80   |   17.60   |   20.70   |   27.40   |   24.40   |   25.90   | [train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_deit_small_gpu8_f16.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_vit_small_infer_mir_f16.sh) |
| ViViM-T (Ours) | 16 |   23.31   |   17.21   |   20.26   |   27.40   |   24.30   |   25.80   |[train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_vivim_tiny_gpu8_f16.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_vivim_tiny_infer_mir_f16.sh) |
| ViViM-S (Ours) | 16 | **26.00** | **19.60** | **22.80** | **28.20** | **25.30** | **26.70** |[train.sh](../scripts/pretrain/run_slurm_pretrain_bs512_vivim_small_gpu8_f16.sh) | [infer.sh](../scripts/mir_zs/run_slurm_lavila_pretrain_bs512_vivim_small_infer_mir_f16.sh) |

### 2. Long-term Video Question-Answer

### 3. Finetuned Multi-instance Retrieval on EK100

| Method                | #Frame | Frame Sampling | Acc (full set) | Train shell | Infer shell |
|-----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|



### 3. Finetuned Multi-instance Retrieval on EK100

| Method                | #Frame |  V2T mAP |  T2V mAP |  Avg mAP | V2T nDCG | T2V nDCG | Avg nDCG | Train shell | Infer shell |
|-----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| TimeSformer (Vanilla) | 4->16 |  52.2  |   44.2   |   48.2   |   64.0   |   61.5   |   62.7   | [train.sh](../scripts/mir_ft/run_slurm_finetune_cls_bs512_timesformer.sh) | [infer.sh](../scripts/mir_ft/run_slurm_finetune_cls_bs512_timesformer_infer_cls_f16.sh) |
| TimeMamba (Ours)   | 4->16 |   **52.4**   |   **45.4**   |   **48.9**   |   **65.9**   |   **63.3**   |   **64.6**   | [train.sh](../scripts/mir_ft/run_slurm_finetune_mir_bs512_timemamba_like_timesformer.sh) | [infer.sh](../scripts/mir_ft/run_slurm_finetune_mir_bs512_timemamba_like_timesformer_infer_mir_f16.sh) |

### 4. Finetuned Action Recognition

#### 4.1 EK100
| Method                | #Frame |Verb Top1 | Noun Top1 | Action Top1 | Action Top5 | Train shell | Infer shell |
|-----------------------|:---------:|:---------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| TimeSformer (Vanilla) | 4->16 |    65.2   |    55.0   |     44.5    |     62.4    | [train.sh](../scripts/cls_ft/run_slurm_finetune_cls_bs512_timesformer.sh) | [infer.sh](../scripts/cls_ft/run_slurm_finetune_cls_bs512_timesformer_infer_cls_f16.sh) |
| TimeMamba (Ours)   | 4->16 |  **68.5** |  **55.8** |   **46.1**  |   **63.8**  | [train.sh](../scripts/cls_ft/run_slurm_finetune_cls_bs512_timemamba_like_timesformer.sh) | [infer.sh](../scripts/cls_ft/run_slurm_finetune_cls_bs512_timemamba_like_timesformer_infer_cls_f16.sh) |