# Video Mamba Suite: Action Recognition


## Main Results

### Kinetics-400

| Method  | #Frame |  Top-1 Acc| Top-5 Acc | shell |
|---------|:----:|:---------:|:---------:|:---------:|
| ViViM-T (Ours) | 16 |   77.51   |   93.27   |  [script.sh](./exp/k400/vivim_tiny/run_f16x224.sh) | 
| ViViM-S (Ours) | 16 |   80.47   |   94.75   |  [script.sh](./exp/k400/vivim_small/run_f16x224.sh) | 

## Model ZOO

You can find all the models and the scripts in [MODEL_ZOO](./docs/MODEL_ZOO.md).

## Usage


For finetuning, you can simply run the fine-tuning scripts as follows:
```shell
bash ./exp/k400/vivim_tiny/run_f16x224.sh
bash ./exp/k400/vivim_small/run_f16x224.sh
```

> **Notes:**
> 1. Chage `DATA_PATH` And `PREFIX` to your data path before running the scripts.
> 2. Set `--finetune` when using masked pretrained model.
> 3. The best checkpoint will be automatically evaluated with `--test_best`.
> 4. Set `--test_num_segment` and `--test_num_crop` for different evaluation strategies.
> 5. To only run evaluation, just set `--eval`.