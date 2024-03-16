# Video Mamba Suite: Action-Anticipation
Implementation for Mamba-based TeSTra (ECCV 2022) 

This codebase is built up on action anticipation of [TeSTRa](https://arxiv.org/abs/2209.09236).


We use TSN as backbones for feature extraction. The details can be found in [TeSTRa](https://arxiv.org/abs/2209.09236) paper.

**NOTICE: Since Action Anticipation empolys causal Mamba, so it uses the vanilla [Mamba](https://github.com/state-spaces/mamba). It is necessary to create a new virtual environment.**


## Results 

**EK100**

|Method | Block | Overall Verb | Overall Noun | Overall Action |
|:----:|:-----:|:----------------:|:-------:|:-------:|
|TeSTra | short Attn | 25.1 | 30.8 | 14.1 |
|TeSTra | short Mamba | **27.9** | **34.1** | **15.2** |

## Installation
* Follow [INSTALL.md](./INSTALL.md) for installing necessary dependencies and compiling the code.
## Data Preparation
### 1. Download EK100 Features and Annotations

We use the video feature and annotations provided by [TeSTra](https://github.com/zhaoyue-zephyrus/TeSTra).

```bash
wget https://utexas.box.com/shared/static/kypifujsplkg0ud7q955amgvoxflqzx5.zip -O rgb_kinetics_bninception.zip
wget https://utexas.box.com/shared/static/2aga6r29o4zdziog3y89aliauguiqhmn.zip -O flow_kinetics_bninception.zip
wget https://utexas.box.com/shared/static/xi1xowkhlmi079suwwq6dlez44lb846e.zip -O target_perframe.zip
wget https://utexas.box.com/shared/static/e9yes31rblmuzb5mdrf3gy1mb7af7a63.zip -O verb_perframe.zip
wget https://utexas.box.com/shared/static/vmg478wjbcf83wc0adw0t9yxduxjqna9.zip -O noun_perframe.zip
unzip rgb_kinetics_bninception.zip -d rgb_kinetics_bninception/ && rm rgb_kinetics_bninception.zip
unzip flow_kinetics_bninception.zip -d flow_kinetics_bninception/ && rm flow_kinetics_bninception.zip
unzip target_perframe.zip -d target_perframe/ && rm target_perframe.zip
unzip verb_perframe.zip -d verb_perframe/ && rm verb_perframe.zip
unzip noun_perframe.zip -d noun_perframe/ && rm noun_perframe.zip
```


### 2.Data Structure

```bash
$YOUR_PATH_TO_EK_DATASET
├── rgb_kinetics_bninception/
|   ├── P01_01.npy (of size L x 2048)
│   ├── ...
├── flow_kinetics_bninception/
|   ├── P01_01.npy (of size L x 2048)
|   ├── ...
├── target_perframe/
|   ├── P01_01.npy (of size L x 3807)
|   ├── ...
├── noun_perframe/
|   ├── P01_01.npy (of size L x 301)
|   ├── ...
├── verb_perframe/
|   ├── P01_01.npy (of size L x 98)
|   ├── ...
```

​	Create softlinks of datasets:
```bash
ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
ln -s $YOUR_PATH_TO_EK_DATASET data/EK100
```



## Training and Evaluation
*First, replace "path/to/repo" in sys.path.append in [train_net.py](tools/train_net.py),  [test_net.py](tools/test_net.py), and [perframe_det_batch_inference.py](src/rekognition_online_action_detection/engines/base_inferences/perframe_det_batch_inference.py) with your own path.*

**Training**

```bash
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
# Finetuning from a pretrained model
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT
```

**Online Inference**

```bash
python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT
```


## Acknowledgement

The codebase is based on [TeSTra](https://github.com/zhaoyue-zephyrus/TeSTra).
We thanks the authors for their efforts.