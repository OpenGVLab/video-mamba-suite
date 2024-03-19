# Video Mamba Suite: Video Temporal Grounding & Highlight Detection

Implementation for Mamba-based UniVTG (ICCV 2023) 
[[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_UniVTG_Towards_Unified_Video-Language_Temporal_Grounding_ICCV_2023_paper.html)



## Results

**QvHighlight**

|Method | Block | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP@Avg | HD@mAP | HD@HIT@1 |
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
UniVTG | Attn | 59.87 | 44.32 | 59.09 | 40.25 | 38.48 | 39.22 | 64.71 |
Ours | ViM | 65.55 | 50.77 | 63.97 | 46.11 | 44.74 | **40.22** | 64.26 |
Ours | DBM | **66.65** | **52.19** | **64.37** | **46.68** | **45.18** | 40.18 | **64.77** |

**Charade-STA**

|Method | Block | R1@0.3 | R1@0.5 | R1@0.7 |
|:----:|:-----:|:----------------:|:-------:|:-------:|
UniVTG | Attn | **68.20** | **57.26** | 34.68 |
Ours | ViM | 68.12 | 57.07 | 35.83 |
Ours | DBM | 68.06 | 57.18 | **36.05** |

## ⚙️ Preparation

Please find instructions in [install.md](install.md) to setup environment and datasets.


## 🚀 Training & Inference

### Downstream (single-gpu)
*Indicate `--resume` to init model by pretraining weight. Refer to our model zoo for detailed parameter settings*

Training: `bash scripts/qvhl_pretrain_mamba.sh`


*Indicate `--eval_init` and `--n_epoch=0` to evaluate selected checkpoint `--resume`.*

Inference: `bash scripts/qvhl_inference_mamba.sh`

You can specify `mamba_type` to switch `ViM` or `DBM` block.



## Acknowledgement

The codebase is based on [UniVTG](https://github.com/showlab/UniVTG).
We thanks the authors for their efforts.
