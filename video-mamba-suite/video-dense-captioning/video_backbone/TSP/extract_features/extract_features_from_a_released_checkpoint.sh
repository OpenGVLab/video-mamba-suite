#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

DATA_PATH= # path/to/video/folder
METADATA_CSV_FILENAME= # path/to/metadata/csv/file. Use the ones provided in the data folder.

##############################
### RELEASED GITHUB MODELS ###
##############################
## main TSP models ->
# r2plus1d_34-tsp_on_activitynet (default)
# r2plus1d_34-tsp_on_thumos14
#
## main TAC baseline models ->
# r2plus1d_34-tac_on_activitynet
# r2plus1d_34-tac_on_thumos14
# r2plus1d_34-tac_on_kinetics
#
## other models from the GVF and backbone architecture ablation studies ->
# r2plus1d_34-tsp_on_activitynet-avg_gvf
# r2plus1d_34-tsp_on_activitynet-no_gvf
# r2plus1d_18-tsp_on_activitynet
# r2plus1d_18-tac_on_activitynet
# r2plus1d_18-tac_on_kinetics
# r3d_18-tsp_on_activitynet
# r3d_18-tac_on_activitynet
# r3d_18-tac_on_kinetics
RELEASED_CHECKPOINT=r2plus1d_34-tsp_on_activitynet # choose one of the models above

# Choose the stride between clips, e.g. 16 for non-overlapping clips and 1 for dense overlapping clips
STRIDE=16 

# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda:0

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$METADATA_CSV_FILENAME" ]; then
    echo "METADATA_CSV_FILENAME variable is not set."
    echo "We provide metadata CSV files for ActivityNet and THUMOS14 in the data folder."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################

OUTPUT_DIR=output/${RELEASED_CHECKPOINT}_features/stride_${STRIDE}/

source activate tsp
mkdir -p $OUTPUT_DIR

python extract_features.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--released-checkpoint $RELEASED_CHECKPOINT \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $OUTPUT_DIR
