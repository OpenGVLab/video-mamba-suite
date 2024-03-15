# Video Mamba Suite: Video Temporal Grounding & Highlight Detection

Implementation for Mamba-based UniVTG (ICCV 2023) 
[[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_UniVTG_Towards_Unified_Video-Language_Temporal_Grounding_ICCV_2023_paper.html)


## ⚙️ Preparation

Please find instructions in [install.md](install.md) to setup environment and datasets.


## 🚀 Training & Inference
### Pretraining (multi-gpu)

Large-scale pretraining: `bash scripts/pretrain.sh`

Multi-datasets co-training: `bash scripts/cotrain.sh`

### Downstream (single-gpu)
*Indicate `--resume` to init model by pretraining weight. Refer to our model zoo for detailed parameter settings*

Training: `bash scripts/qvhl_pretrain.sh`


*Indicate `--eval_init` and `--n_epoch=0` to evaluate selected checkpoint `--resume`.*

Inference: `bash scripts/qvhl_inference.sh`



## Acknowledgement

The codebase is based on [UniVTG](https://github.com/showlab/UniVTG).
We thanks the authors for their efforts.
