# Model Zoo

## Pre-training LaViLa

### 
<details><summary> Train a baseline dual-encoder with ViT-B </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 \
    scripts/main_lavila_pretrain.py \
    --root /new-pool/Datasets/Ego4d/v1/videos_288px_15sec/ \
    --root-val datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

<details><summary> Train an LM-augmented dual-encoder (LaViLa) with ViT-B </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 \
    scripts/main_lavila_pretrain.py \
    --root /new-pool/Datasets/Ego4d/v1/videos_288px_15sec/ \
    --root-val datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --train-metadata datasets/Ego4D/ego4d_train.rephraser.no_punkt_top3.pkl \
    --train-metadata-aux datasets/Ego4D/ego4d_train.narrator_63690737.return_10.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

|  Corpus  | LLM-aug. | Corpus size | Backbone | per-gpu<br>batch-size | GPU×hour^ | EK-100 MIR<br>avg. mAP | EK-100 MIR<br>avg. nDCG |                                checkpoint                               | md5sum |
| :------: | :------: | :---------: | :------: | :----------------: | :-------: | :--------------------: | :---------------------: | :---------------------------------------------------------------------: | :----: |
|  Ego4D   |    no    |   4.0M      |  ViT-B   |       256          |  ~130    |       27.5/28.4        |       29.1/29.5         | [best Epoch](https://utexas.box.com/s/yp1krj3dsmr8wj0sz01t10bwa9fgq3zy) | fc3b7f |
|  Ego4D   |    yes   |    35M      |  ViT-B   |       256          |   ~260    |       31.1/32.9        |       31.9/32.7         | [best Epoch](https://utexas.box.com/s/e681nrxivc9makufvrumrfuaopk57h4n) | 91a90b |
|  Ego4D   |    yes   |    35M      |  ViT-L   |       112          |   ~680    |       36.4/37.6        |       35.1/35.3         | [best Epoch](https://utexas.box.com/s/1iatmrs7ufdeooce09a61t1n6wsouf4l) | f377f6 |



^ Hardware configuration: 8x NVIDIA A5000 (24GB) GPUs + 2x Intel Xeon Gold 5220(R) 24-Core CPU @ 2.20GHz (96 threads in total).

## Fine-tuning the video-language dual-encoder on down-stream tasks

### EK-100 Multi-Instance Retrieval (MIR)

<details><summary> Finetune a pretrained dual-encoder on EK-100 MIR </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>


| LLM-aug. | Backbone | V->T mAP | T->V mAP | avg mAP | V->T nDCG | T->V nDCG | avg nDCG |                               checkpoint                                | md5sum |
| :------: | :------: | :------: | :------: | :-----: | :-------: | :-------: | :------: | :---------------------------------------------------------------------: | :----: |
|   yes    |   ViT-B  |   55.7   |   48.2   |  52.0   |   67.8    |   65.3    |   66.5   | [best epoch](https://utexas.box.com/s/ke5kwfixttb4t7uxdbs9gmiiuu1582dg) | e099c0 |
|   yes    |   ViT-L  |   57.9   |   51.1   |  54.5   |   70.4    |   67.6    |   69.0   | [best epoch](https://utexas.box.com/s/m7f65hg9eonz34g0l2x5r0t92ouh0u4w) | f82079 |


### EK-100 Action Recognition (CLS)


<details><summary> Finetune a pretrained dual-encoder on EK-100 CLS </summary>

```bash
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
```

</details>

| LLM-aug. | Backbone | Verb Top1 | Noun Top1 | Action Top1 |                                checkpoint                               | md5sum |
| :------: | :------: | :-------: | :-------: | :---------: | :---------------------------------------------------------------------: | :----: |
|   no     |   ViT-B  |   67.9    |   57.6    |    47.3     | [best epoch](https://utexas.box.com/s/2fkvtc67m0f82wmm5cnqfo7wg951lobv) | b40f3e |
|   yes    |   ViT-B  |   70.0    |   59.4    |    49.5     | [best epoch](https://utexas.box.com/s/8iokob6ahb94gp1bqbmauhpeunqwx79j) | 6c3c5e |
|   yes    |   ViT-L  |   73.0    |   65.4    |    54.4     | [best epoch](https://utexas.box.com/s/crnqo9bu0owtfz4yc1yqf8hz6g0ze39b) | 1871f4 |
