# Video Mamba Suite: Temporal-Action-Localization
Implementation for Mamba-based ActionFormer (ECCV 2022) 

This codebase is built up on temporal action localization of [InternVideo](https://arxiv.org/abs/2212.03191).


We use InternVideo2-1B and InternVideo2-6B from our InternVideo2 as backbones for feature extraction. The details can be found in [InternVideo2](https://arxiv.org/abs/2403.15377) paper (to be released soon)。


## Results 

**THUMOS-14 + InternVideo2-6B**

|Method | Block | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | mAP@Avg |
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:|:-------:|:-------:|
ActionFormer | Win Attn | 82.26 | 81.85 | 75.05 | 65.82 | 50.27 | 71.86 |
ActionMamba | ViM | 86.22 | 82.87 | 76.42 | **66.43** | 50.25 | 72.44 |
ActionMamba | DBM | **86.89** | **83.09** | **76.90** | 65.91 | **50.82** | **72.72** |



**ActivityNet + InternVideo2-6B**

|Method | Block | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP@Avg |
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:
ActionFormer | Win Attn | 61.47 | 44.61 | 12.73 | 41.19 |
ActionMamba | ViM | 62.31 | 43.17 | 9.65 | 41.77 |
ActionMamba | DBM | **62.43** | **43.49** | **10.23** | **42.02** |

**HACS Segment + InternVideo2-6B**

|Method | Block | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP@Avg |
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:
ActionFormer | Win Attn | 62.62 | 44.61 | 12.73 | 43.34 |
ActionMamba | ViM | 63.78 | 45.45 | 13.01 | 44.26 |
ActionMamba | DBM | **64.02** | **45.71** | **13.34** | **44.56** |

**FineAction + InternVideo2-1B**

|Method | Block | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP@Avg |
|:----:|:-----:|:----------------:|:-------:|:-------:|:-------:
ActionFormer | Win Attn | 43.11 | 27.09 | 5.32 | 27.22 |
ActionMamba | ViM | 44.15 | 28.30 | 6.14 | 28.36 |
ActionMamba | DBM | **45.44** | **28.82** | **6.79** | **29.04** | 



## Installation
* Follow [INSTALL.md](./INSTALL.md) for installing necessary dependencies and compiling the code.
## Data Preparation
### 1. Download InternVideo2 Features and Annotations

#### THUMOS-14
**InternVideo-6B feature:** [huggingface](https://huggingface.co/datasets/cg1177/thumos14_internvideo2_6b_w16_s4)

#### ActivityNet
**InternVideo-6B feature:** [huggingface](https://huggingface.co/datasets/cg1177/activitynet_internvideo2_6b_w16_s8)

#### HACS Segment
**InternVideo-6B feature:** [huggingface](https://huggingface.co/datasets/cg1177/hacs_segment_internvideo2_6b_w16_s8)

#### FineAction
**InternVideo-1B feature:** [huggingface](https://huggingface.co/datasets/cg1177/fineaction_internvideo2_1b_w16_s4)


#### Download
For instance, THUMOS-14 InternVideo-6B feature:
```bash
git lfs install
git clone https://huggingface.co/datasets/cg1177/thumos14_internvideo2_6b_w16_s4
```


### 2.Unzip features

For instance, THUMOS-14 InternVideo-6B feature:
```bash
cd thumos14_internvideo2_6b_w16_s4
cat thumos14_internvideo2_6b_w16_s4.part* > thumos14_internvideo2_6b_w16_s4.tar.gz
tar xvf thumos14_internvideo2_6b_w16_s4.tar.gz
```

### 3. Modify the `feat_folder` field in each config file


## Details for Feature: 
The THUMOS-14 & FineAction features are extracted from InternVideo2 models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `4 frames`. This gives one feature vector per `4/30 ~= 0.1333` seconds.

The ANet & HACS features are extracted from Video_MAE models pretrained on Kinetics using clips of `16 frames` at the video frame rate (`~30 fps`) and a stride of `16 frames`. This gives one feature vector per `16/30 ~= 0.5333` seconds.



## Training and Evaluation
* Train the ActionMamba with InternVideo2 features. 
* This will create a experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
bash run_all_thumos_mamba.sh
bash run_all_anet_mamba.sh
bash run_all_hacs_mamba.sh
bash run_all_fineaction_mamba.sh
```




## Acknowledgement

The codebase is based on [ActionFormer](https://github.com/ChinaYi/ASFormer), [InternVideo](https://github.com/OpenGVLab/InternVideo) and [InternVideo2]().
We thanks the authors for their efforts.