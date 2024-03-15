[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tsp-temporally-sensitive-pretraining-of-video/temporal-action-localization-on-activitynet)](https://paperswithcode.com/sota/temporal-action-localization-on-activitynet?p=tsp-temporally-sensitive-pretraining-of-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tsp-temporally-sensitive-pretraining-of-video/temporal-action-proposal-generation-on)](https://paperswithcode.com/sota/temporal-action-proposal-generation-on?p=tsp-temporally-sensitive-pretraining-of-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tsp-temporally-sensitive-pretraining-of-video/dense-video-captioning-on-activitynet)](https://paperswithcode.com/sota/dense-video-captioning-on-activitynet?p=tsp-temporally-sensitive-pretraining-of-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tsp-temporally-sensitive-pretraining-of-video/temporal-action-localization-on-thumos14)](https://paperswithcode.com/sota/temporal-action-localization-on-thumos14?p=tsp-temporally-sensitive-pretraining-of-video)

# TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks

<img align="right" width=40% src="img/tsp.png">

[[Paper]](https://arxiv.org/pdf/2011.11479.pdf)
[[Project Website]](http://humamalwassel.com/publication/tsp/)

This repository holds the source code, pretrained models, and pre-extracted features for the TSP method.

Please cite this work if you find TSP useful for your research.
```
@inproceedings{alwassel_2021_tsp,
  title={TSP: Temporally-Sensitive Pretraining of Video Encoders for Localization Tasks},
  author={Alwassel, Humam and Giancola, Silvio and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International
             Conference on Computer Vision (ICCV) Workshops},
  year={2021}
}
```

## Pre-extracted TSP Features

We provide pre-extracted features for ActivityNet v1.3 and THUMOS14 videos. The feature files are saved in H5 format, where we map each `video-name` to a features tensor of size `N x 512`, where `N` is the number of features and `512` is the feature size. Use `h5py` python package to read the feature files. Not familiar with H5 files or `h5py`? here is a quick start [guide](https://docs.h5py.org/en/stable/).

### For ActivityNet v1.3 dataset
**Download**:
[[train subset]](https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-train_features.h5)
[[valid subset]](https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-valid_features.h5)
[[test subset]](https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-test_features.h5)

**Details**: The features are extracted from the R(2+1)D-34 encoder pretrained with TSP on ActivityNet ([released model](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth)) using clips of `16 frames` at a frame rate of `15 fps` and a stride of `16 frames` (*i.e.,* **non-overlapping** clips). This gives one feature vector per `16/15 ~= 1.067` seconds.


### For THUMOS14 dataset

**Download**:
[[valid subset]](https://github.com/HumamAlwassel/TSP/releases/download/thumos14_features/r2plus1d_34-tsp_on_thumos14-valid_features.h5)
[[test subset]](https://github.com/HumamAlwassel/TSP/releases/download/thumos14_features/r2plus1d_34-tsp_on_thumos14-test_features.h5)

**Details**: The features are extracted from the R(2+1)D-34 encoder pretrained with TSP on THUMOS14 ([released model](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_thumos14-max_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_4-e6a30b2f.pth)) using clips of `16 frames` at a frame rate of `15 fps` and a stride of `1 frame` (*i.e.,* dense **overlapping** clips). This gives one feature vector per `1/15 ~= 0.067` seconds.

## Setup
Clone this repository and create the conda environment.
```
git clone https://github.com/HumamAlwassel/TSP.git
cd TSP
conda env create -f environment.yml
conda activate tsp
```

## Data Preprocessing
Follow the instructions [here](data) to download and preprocess the input data.

## Training
We provide training scripts for the TSP models and the TAC baselines [here](train).

## Feature Extraction
You can extract features from released pretrained models or from local checkpoints using the scripts [here](extract_features).

**Acknowledgment**: Our source code borrows implementation ideas from [pytorch/vision](https://github.com/pytorch/vision) and [facebookresearch/VMZ](https://github.com/facebookresearch/VMZ) repositories.
