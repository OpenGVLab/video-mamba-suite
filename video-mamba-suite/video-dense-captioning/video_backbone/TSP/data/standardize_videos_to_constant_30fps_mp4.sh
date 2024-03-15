#!/bin/bash
#
# Script to convert all videos in <input_folder> to mp4 videos with constant frame rate of 30fps.
# The output videos are saved in <output_folder>.
#
# usage: bash standardize_video_to_constant_30fps_mp4.sh <input_folder> <output_folder>

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "usage: bash standardize_video_to_constant_30fps_mp4.sh <input_folder> <output_folder>"
    exit 1
fi


INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

echo "INPUT_FOLDER=$INPUT_FOLDER"
echo "OUTPUT_FOLDER=$OUTPUT_FOLDER"

mkdir -p $OUTPUT_FOLDER

for input_video_path in $INPUT_FOLDER/*;
do
	video_filename=$(basename $input_video_path)
	video_name="${video_filename%.*}"
	output_video_path="$OUTPUT_FOLDER/$video_name.mp4"

	echo "ffmpeg -y -i $input_video_path -filter:v fps=fps=30 $output_video_path"
	ffmpeg -y -i $input_video_path -filter:v fps=fps=30 $output_video_path
done
