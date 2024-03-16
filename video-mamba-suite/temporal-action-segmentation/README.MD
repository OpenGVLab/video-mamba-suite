# Video Mamba Suite: Temporal Action Segmentation
Implementation for Mamba-based ASFormer (BMVC 2021) 
[[paper]](https://arxiv.org/pdf/2110.08568.pdf)

## Preparation
1. Install pytorch and dependencies. The code is run successfully on (1) torch==1.13.1+cu117 or (2) torch==2.1.2+cu118, other pytorch versions may also work. 
```bash
conda install -y conda-forge::seaborn
conda install -y conda-forge::matplotlib
```

2. Install mamba follow the main README.md, make sure you can import mamba successfully
```bash
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_new import Mamba 
```

### Download Video Features
Follow the data preparation steps of ASFormer for preparing features of 50Salads, Breakfast and GTEA datasets.
```bash
1. Download the dataset data.zip at (https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) or (https://zenodo.org/record/3625992#.Xiv9jGhKhPY). 
2. Unzip the data.zip file to the current folder. There are three datasets in the ./data folder, i.e. ./data/breakfast, ./data/50salads, ./data/gtea
```

## Train your own model
You can train the model by yourself with following command.
```
Run python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4/5 --mamba
Run python main.py --action=predict --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
Run python eval.py --dataset=50salads/gtea/breakfast --split=0/1/2/3/4/5 to evaluate the performance. **NOTE**: split=0 will evaulate the average results for all splits, It needs to be done after you complete all split predictions.
```

## Acknowledgement

The codebase is based on [ASFormer](https://github.com/ChinaYi/ASFormer).
We thanks the authors for their efforts.