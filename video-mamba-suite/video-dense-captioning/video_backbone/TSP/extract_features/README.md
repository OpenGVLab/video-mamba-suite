# TSP Feature Extraction

Follow the data preprocessing instructions described [here](../data) before extracting features. We provide scripts for feature extraction using the released pretrained models or using a local checkpoint.

### From Released Pretrained Models
Use the `extract_features_from_a_released_checkpoint.sh` script to extract features from the official released models. You need to manually set the following variables:
- `DATA_PATH`: Path to the video folder.
- `METADATA_CSV_FILENAME`: Path to a metadata CSV file. For ActivityNet and THUMOS14, use the CSV files precomputed in the [data](../data) folder. If you want to extract features for other video datasets, first standardized the videos and then generate the metadata files as per the instructions [here](../data), specifically step 2 and 4.
- `RELEASED_CHECKPOINT`: Name of the one of the `13` released pretrained model. Refer to the tables below for more details.
- `STRIDE`: Choose the stride between clips, *e.g.,* `16` for non-overlapping clips and `1` for dense overlapping clips.
- (Optional) `SHARD_ID`, `NUM_SHARDS`, `DEVICE`: Split the videos in the CSV into multiple shards for parallel feature extraction. Increase the number of shards and run the script independently on separate GPU devices, each with a different `SHARD_ID` from `0` to `NUM_SHARDS-1`. Each shard will process `num_videos / NUM_SHARDS` videos.

### From a Local Checkpoint
Use the `extract_features_from_a_local_checkpoint.sh` script to extract features from a local checkpoint. You need to manually set the same variables above plus the following 2 variables instead of `RELEASED_CHECKPOINT`:
- `LOCAL_CHECKPOINT`: Path to the local checkpoint `.pth` file.
- `BACKBONE`: The backbone used in the local checkpoint: `r2plus1d_34`, `r2plus1d_18`, or `r3d_18`.

## Post Processing Output
The feature extraction script will output a `.pkl` file for each video. Merge all the `.pkl` files into one `.h5` file as follows:

```
python merge_pkl_files_into_one_h5_feature_file.py --features-folder <path/to/feature/output/folder/> --output-h5 <features_filenames.h5>
```

------

**Released Pretrained Models**

**Main TSP models**
| Name                                     | Description                                                 | Weights |
| ---------------------------------------- | ----------------------------------------------------------- | ------- |
| `r2plus1d_34-tsp_on_activitynet`         | R(2+1)D-34 pretrained with TSP on ActivityNet               | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth) |
| `r2plus1d_34-tsp_on_thumos14`            | R(2+1)D-34 pretrained with TSP on THUMOS14                  | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_thumos14-max_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_4-e6a30b2f.pth) |

**Main TAC baseline models**
| Name                                     | Description                                                 | Weights |
| ---------------------------------------- | ----------------------------------------------------------- | ------- |
| `r2plus1d_34-tac_on_activitynet`         | R(2+1)D-34 pretrained with TAC on ActivityNet               | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.002-epoch_5-98ccac94.pth) |
| `r2plus1d_34-tac_on_thumos14`            | R(2+1)D-34 pretrained with TAC on THUMOS14                  | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_thumos14-backbone_lr_0.00001-fc_lr_0.002-epoch_3-54b5c8aa.pth) |
| `r2plus1d_34-tac_on_kinetics`            | R(2+1)D-34 pretrained with TAC on Kinetics                  | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_kinetics-0547130e.pth) |

**Other models from the GVF and backbone architecture ablation studies**
| Name                                     | Description                                                 | Weights |
| ---------------------------------------- | ----------------------------------------------------------- | ------- |
| `r2plus1d_34-tsp_on_activitynet-avg_gvf` | R(2+1)D-34 pretrained with TSP on ActivityNet (average GVF) | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-avg_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-8b74eaa2.pth) |
| `r2plus1d_34-tsp_on_activitynet-no_gvf`  | R(2+1)D-34 pretrained with TSP on ActivityNet (without GVF) | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-no_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-fb38fdd2.pth) |
| `r2plus1d_18-tsp_on_activitynet`         | R(2+1)D-18 pretrained with TSP on ActivityNet               | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-22835b73.pth) |
| `r2plus1d_18-tac_on_activitynet`         | R(2+1)D-18 pretrained with TAC on ActivityNet               | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.004-epoch_5-9f56941a.pth) |
| `r2plus1d_18-tac_on_kinetics`            | R(2+1)D-18 pretrained with TAC on Kinetics                  | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_kinetics-76ce975c.pth) |
| `r3d_18-tsp_on_activitynet`              | R3D-18 pretrained with TSP on ActivityNet                   | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-85584422.pth) |
| `r3d_18-tac_on_activitynet`              | R3D-18 pretrained with TAC on ActivityNet                   | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_activitynet-backbone_lr_0.001-fc_lr_0.01-epoch_5-31fd6e95.pth) |
| `r3d_18-tac_on_kinetics`                 | R3D-18 pretrained with TAC on Kinetics                      | [checkpoint](https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_kinetics-dcd952c6.pth) |

