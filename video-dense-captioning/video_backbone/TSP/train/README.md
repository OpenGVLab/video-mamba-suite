# TSP Training

We provide four training scripts:
- `train_tsp_on_activitynet.sh`: pretraining R(2+1)D-34 encoder with TSP on ActivityNet
- `train_tsp_on_thumos14.sh`: pretraining R(2+1)D-34 encoder with TSP on THUMOS14
- `train_tac_on_activitynet.sh`: pretraining R(2+1)D-34 encoder with TAC on ActivityNet (baseline)
- `train_tac_on_thumos14.sh`: pretraining R(2+1)D-34 encoder with TAC on THUMOS14 (baseline)

## Launching the Training Scripts

Before launching each script, you need to manually set **3 variables** inside each file:
- `ROOT_DIR`: The root directory of either the ActivityNet or THUMOS14 videos. Follow the data preprocessing instructions and subfolders naming described [here](../data).
- `NUM_GPUS`: The number of GPUs to use for training. We used 2 V100 (32G) GPUs in our TSP experiments, but the code is generic and can be run on any number of GPUs.
- `DOWNSCALE_FACTOR`: The default batch size and learning rates were optimized for a GPU with 32G memory. We understand that such GPUs might not be accessible to all of the community. Thus, the training code can seamlessly be adapt to run on a smaller GPU memory size by adjusting this variable. Set `DOWNSCALE_FACTOR` to `1`, `2`, or `4` if you have a GPU with 32G, 16G, or 8G memory, respectively. The script will automatically downscale the batch size and the learning rate accordingly to keep the same expected performance.

## Experiment Output

- Checkpoint per epoch (*e.g.,* `epoch_3.pth`): a `.pth` file containing the state dictionary of the model, optimizer, and learning rate scheduler. The checkpoint files can be used to resume the training (use `--resume` and `--start-epoch` input parameters in `train.py`) or to extract features (use the scripts [here](../extract_features)).
- Metric results file (`results.txt`): A log of the metrics results on the validation subset after each epoch. We choose the best pretrained model based on the epoch with the highest `Avg Accuracy` value.

## Interested in Reproducing the Ablation Studies?

Train with different encoder architectures? Change the variable `BACKBONE` to either `r2plus1d_18` or `r3d_18`.
Train without GVF? Remove the line `--global-video-features $GLOBAL_VIDEO_FEATURES \` from the `train.py` call at the end.
Train with average GVF? Set `GLOBAL_VIDEO_FEATURES=../data/activitynet/global_video_features/r2plus1d_34-avg_gvf.h5`.
Train with only the temporal region classification head? Set `LABEL_COLUMNS=temporal-region-label` and `LABEL_MAPPING_JSONS=../data/activitynet/activitynet_v1-3_temporal_region_label_mapping.json`. Finally, make sure to rename `OUTPUT_DIR` to avoid overwriting previous experiment when reproducing the ablation studies.
