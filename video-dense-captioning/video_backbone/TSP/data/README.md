# Data Preprocessing

**Step 1**: Download the ActivityNet v1.3 and THUMOS14 videos. For ActivityNet, you can submit a data request [here](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform). For THUMOS14, you can download it directly from the [official website](http://crcv.ucf.edu/THUMOS14/download.html).

**Step 2**: Standardize all videos to MP4 format with a constant frame rate of 30fps using the script `standardize_videos_to_constant_30fps_mp4.sh`:
```
bash standardize_video_to_constant_30fps_mp4.sh <input_folder> <output_folder>
```

**Step 3**: Split the ActivityNet videos into three subfolders: `train` (10024 videos), `valid` (4926 videos), and `test` (5044 videos) using the official splits. Similarly, split THUMOS14 into `valid` (200 videos) and `test` (213 videos) subfolders.

**Step 4**: Generate metadata CSV files for each ActivityNet and THUMOS14 subset using the script `generate_metadata_csv.py`. _This step is already pre-computed for the standardized ActivityNet and THUMOS14 videos and saved in the `activitynet` and `thumos14` folders_.
```
python generate_metadata_csv.py --video-folder <path/to/folder> --output-csv <output_filename.csv>
```
