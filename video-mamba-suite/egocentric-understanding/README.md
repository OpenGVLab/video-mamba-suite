# :airplane: avion
AVION is short for A VIdeo model in ONe day. AVION (meaning plane in French and Spanish) is fast.


[**Training a Large Video Model on a Single Machine in a Day**](http://arxiv.org/abs/2309.16669)  
Yue Zhao, Philipp Kr&auml;henb&uuml;hl  
UT Austin  
[arxiv](http://arxiv.org/abs/2309.16669) | [bibtex](#citing-avion) 


## Installation

See [INSTALL.md](docs/INSTALL.md) to install this code.


## Main results

1. AVION enables video-language contrastive pre-training on Ego4D (original narratives) **on a single node of 8× consumer-grade GPUs within a day**.

    | Method | Backbone | batch-size<br>per GPU | GPU memory | Hardware | GPU×hour^ | EK100 MIR<br>0-shot Avg. mAP | 
    | :----: | :------: | :-------------------: | :--------: | :------: | :-------: | :--------------------------: |
    | EgoVLP |   TSF-B  |          16           |     22     | 32× A100 |   1536    |            22.1              |
    |  Ours  |   ViT-B  |         256           |     19     | 8× A5000 |    130    |            27.4              |

    ^The reported GPU×hour is *not* normalized for GPU generations. The cost for EgoVLP is obtained from the [original paper](https://arxiv.org/abs/2206.01670) (Sec 6.1).

2. AVION speeds up LLM-augmented video-language contrastive pre-training (LaViLa) on Ego4D.

    a. Pretraining cost and performance.

    | Method | Backbone | batch-size<br>per GPU | GPU memory | Hardware | GPU×hour^ | EK100 MIR<br>0-shot Avg. mAP |
    | :----: | :------: | :-------------------: | :--------: | :------: | :-------: | :--------------------------: |
    | LaViLa |   TSF-B  |          32           |     25     | 32× V100 |   1824    |            30.9              |
    |  Ours  |   ViT-B  |         256           |     19     | 8× A5000 |    260    |            33.2              |

    ^The reported GPU×hour is *not* normalized for GPU generations.

    b. Downstream performance.

    | Method | Backbone | EK100 MIR<br>Avg. mAP | EK100 MIR<br>Avg. nDCG | EK100 CLS<br>Action Top-1 |
    | :----: | :------: | :-------------------: | :--------------------: | :-----------------------: |
    | LaViLa |  TSF-B   |          50.5         |          65.0          |          46.9             |
    |  Ours  |  ViT-B   |          51.7         |          66.8          |          49.5             |
    | LaViLa |  TSF-L   |          50.9         |          66.5          |          51.0             |
    |  Ours  |  ViT-L   |          54.5         |          69.0          |          54.5             |

    :trophy: LaViLa+AVION helps us win [CVPR 2023 EPIC-Kitchens Challenges](https://epic-kitchens.github.io/2023#results) in both Action Recognition and Multi-Instance Retrieval Tasks by a significant margin.

3. AVION speeds up VideoMAE pre-training.

    |  Method  | Backbone | Epochs | GPU×hour^^ | top-1/top-5 (w/. FT) |
    | :------: | :------: | :----: | :--------: | :------------------: |
    | VideoMAE |  ViT-B   |  800   |    995     |      80.0/94.4       |
    |   Ours   |  ViT-B   |  800   |    583     |      80.1/94.5       |

    ^^Both GPU×hour are measured on the same hardware environment (4× A5000 GPU).

For more details, please refer to [MODEL_ZOO](./docs/MODEL_ZOO.md).

## License

[MIT License](./LICENSE).


## Acknowledgements

* The vision-language contrastive pretraining part is refactored from [LaViLa](https://github.com/facebookresearch/LaViLa).
* The MAE-style self-supervised pre-training part is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE/).



## Citing AVION

```bibtex
@article{zhao2023training,
  title={Training a large video model on a single machine in a day},
  author={Zhao, Yue and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:2309.16669},
  year={2023}
}
```

```bibtex
@inproceedings{zhao2023lavila,
  title={Learning Video Representations from Large Language Models},
  author={Zhao, Yue and Misra, Ishan and Kr{\"a}henb{\"u}hl, Philipp and Girdhar, Rohit},
  booktitle={CVPR},
  year={2023}
}
```
