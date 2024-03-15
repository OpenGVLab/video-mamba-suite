# TSP features from https://github.com/HumamAlwassel/TSP
# download the following files and reformat them into data/features/tsp/VIDEO_ID.npy where VIDEO_ID starts with 'v_'
wget https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-train_features.h5
wget https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-valid_features.h5
wget https://github.com/HumamAlwassel/TSP/releases/download/activitynet_features/r2plus1d_34-tsp_on_activitynet-test_features.h5
python convert_tsp_h5_to_npy.py
