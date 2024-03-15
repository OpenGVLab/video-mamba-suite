#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

ROOT_DIR=
NUM_GPUS=

# Choose the appropriate batch size downscale factor for your GPU memory size
# DOWNSCALE_FACTOR=1 --> a 32G memory GPU (default)
# DOWNSCALE_FACTOR=2 --> a 16G memory GPU
# DOWNSCALE_FACTOR=4 --> a 8G memory GPU
DOWNSCALE_FACTOR=1

if [ -z "$ROOT_DIR" ]; then
    echo "ROOT_DIR variable is not set."
    echo "Please set ROOT_DIR to the location of the THUMOS14 videos."
    echo "The directory must contain two subdirectories: valid and test"
    exit 1
fi

if [ -z "$NUM_GPUS" ]; then
    echo "NUM_GPUS variable is not set."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################

TRAIN_SUBDIR=valid
VALID_SUBDIR=test
TRAIN_CSV_FILENAME=../data/thumos14/thumos14_valid_tsp_groundtruth.csv
VALID_CSV_FILENAME=../data/thumos14/thumos14_test_tsp_groundtruth.csv
LABEL_COLUMNS="action-label temporal-region-label"
LABEL_MAPPING_JSONS="../data/thumos14/thumos14_action_label_mapping.json \
                     ../data/thumos14/thumos14_temporal_region_label_mapping.json"
LOSS_ALPHAS="1.0 1.0"
GLOBAL_VIDEO_FEATURES=../data/thumos14/global_video_features/r2plus1d_34-max_gvf.h5

BACKBONE=r2plus1d_34

BATCH_SIZE=32
BACKBONE_LR=0.0001
FC_LR=0.004

OUTPUT_DIR=output/${BACKBONE}-tsp_on_thumos14/backbone_lr_${BACKBONE_LR}-fc_lr_${FC_LR}/

MY_MASTER_ADDR=127.0.0.1
MY_MASTER_PORT=$(shuf -i 30000-60000 -n 1)

# downscaling
BATCH_SIZE=$(bc <<< $BATCH_SIZE/$DOWNSCALE_FACTOR)
BACKBONE_LR=$(bc -l <<< $BACKBONE_LR/$DOWNSCALE_FACTOR)
FC_LR=$(bc -l <<< $FC_LR/$DOWNSCALE_FACTOR)

source activate tsp
mkdir -p $OUTPUT_DIR
export OMP_NUM_THREADS=6

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
--master_addr $MY_MASTER_ADDR --master_port $MY_MASTER_PORT --use_env \
train.py \
--root-dir $ROOT_DIR \
--train-subdir $TRAIN_SUBDIR \
--valid-subdir $VALID_SUBDIR \
--train-csv-filename $TRAIN_CSV_FILENAME \
--valid-csv-filename $VALID_CSV_FILENAME \
--label-mapping-jsons $LABEL_MAPPING_JSONS \
--label-columns $LABEL_COLUMNS \
--loss-alphas $LOSS_ALPHAS \
--global-video-features $GLOBAL_VIDEO_FEATURES \
--backbone $BACKBONE \
--batch-size $BATCH_SIZE \
--backbone-lr $BACKBONE_LR \
--fc-lr $FC_LR \
--output-dir $OUTPUT_DIR \
