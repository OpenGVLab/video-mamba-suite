# download i3d features (rgb+flow) and vggish features of ActivityNet Captions
# Modified from https://github.com/v-iashin/BMT/blob/master/download_data.sh
# Copyright (c) 2020 Vladimir Iashin


# checking if wget is installed on a computer
if ! command -v wget &> /dev/null
then
    echo "wget: command not found"
    echo ""
    echo "wget command could not be found on your computer. Please, install it first."
    echo "If you cannot/dontwantto install wget, you may try to download the features manually."
    echo "You may find the links and correct paths in this file."
    echo "Make sure to check the md5 sums after manual download:"
    echo "./data/i3d_25fps_stack64step64_2stream_npy.zip    d7266e440f8c616acbc0d8aaa4a336dc"
    echo "./data/vggish_npy.zip    9a654ad785e801aceb70af2a5e1cffbe"
    echo "./.vector_cache/glove.840B.300d.zip    2ffafcc9f9ae46fc8c95f32372976137"
    exit
fi


echo "Downloading i3d features"
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/i3d_25fps_stack64step64_2stream_npy.zip -q --show-progress
echo "Downloading vggish features"
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/vggish_npy.zip -q --show-progress

#echo "Downloading GloVe embeddings"
#mkdir .vector_cache
#cd .vector_cache
#wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/glove.840B.300d.zip  -q --show-progress
#cd ../

echo "Checking for correctness of the downloaded files"

i3d_md5=($(md5sum ./data/i3d_25fps_stack64step64_2stream_npy.zip))
if [ "$i3d_md5" == "d7266e440f8c616acbc0d8aaa4a336dc" ]; then
    echo "OK: i3d features"
else
    echo "ERROR: .zip file with i3d features is corrupted"
    exit 1
fi

vggish_md5=($(md5sum ./data/vggish_npy.zip))
if [ "$vggish_md5" == "9a654ad785e801aceb70af2a5e1cffbe" ]; then
    echo "OK: vggish features"
else
    echo "ERROR: .zip file with vggish features is corrupted"
    exit 1
fi

glove_md5=($(md5sum ./.vector_cache/glove.840B.300d.zip))
if [ "$glove_md5" == "2ffafcc9f9ae46fc8c95f32372976137" ]; then
    echo "OK: glove embeddings"
else
    echo "ERROR: .zip file with glove embeddings is corrupted"
    exit 1
fi

echo "Unpacking i3d (~1 min)"

unzip -q i3d_25fps_stack64step64_2stream_npy.zip
echo "Unpacking vggish features"
unzip -q vggish_npy.zip

echo "Done"