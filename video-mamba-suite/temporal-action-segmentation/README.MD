# Video Mamba Suite: Temporal Action Segmentation
Implementation for Mamba-based ASFormer (BMVC 2021) 
[[paper]](https://arxiv.org/pdf/2110.08568.pdf)

## Train your own model
Also, you can retrain the model by yourself with following command.
```
python main.py --action=train --dataset=50salads/gtea/breakfast --split=1/2/3/4/5
```
The training process is very stable in our experiments. It convergences very fast and is not sensitive to the number of training epochs.


## Acknowledgement

The codebase is based on [ASFormer](https://github.com/ChinaYi/ASFormer).
We thanks the authors for their efforts.

