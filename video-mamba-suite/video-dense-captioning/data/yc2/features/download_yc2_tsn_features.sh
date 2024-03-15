http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz

# Download TSN feature files for the youcook2 dataset, refer to https://github.com/salesforce/densecap#data-preparation for more details about feature extraction.
wget http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz
wget http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/testing_feat_yc2.tar.gz

tar xvzf training_feat_yc2.tar.gz
tar xvzf validation_feat_yc2.tar.gz
tar xvzf testing_feat_yc2.tar.gz
mkdir resnet_bn
mv testing/* resnet_bn
mv training/* resnet_bn
mv validation/* resnet_bn
