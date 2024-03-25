# Installation

## Clone the repo

Clone the repo recursively. Note that it is important to add the `--recursive` option.
```bash
git clone --recursive git@github.com:zhaoyue-zephyrus/avion.git
```

## Requirements
The repo has been developed on a Ubuntu (22.04) system with 8x NVIDIA RTX A5000 (24GB) GPUs. We use CUDA 11.7 and pytorch 1.13.1.

## Example conda environment setup
```bash
conda create --name avion python=3.10 -y
conda activate avion
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install ninja==1.11.1  # install ninja first for building flash-attention
CUDA_HOME=$CUDA_HOME pip install -r requirements.txt
```

## Compile Decord
We build the Fused DecodeCrop operator on top of [dmlc/decord](https://github.com/dmlc/decord). To install our forked decord from source:

(1) Build the shared library
```bash
cd third_party/decord/
mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make
```

(2) Install python bindings:
```bash
cd ../python
python3 setup.py install --user
```

(3) Check the installation is correct:

First, add the python path to $PYTHONPATH with
```bash
PYTHONPATH=$PYTHONPATH:$PWD
```

Next, return to `<YOUR_AVION_HOME_PATH>` and run 
```bash
python -c "import decord; print(decord.__path__)"
```
It should print out the decord build path `['<YOUR_AVION_HOME_PATH>/third_party/decord/python/decord']`.


### Troubleshooting:

* If you see an empty folder under `third_party/decord/` in step (1), it indicates that you did not clone the repo recursively. You can run `git submodule update --init --recursive` to fetch the submodule.

* If you see ```OSError: <YOUR_CONDA_HOME_PATH>/envs/avion/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by <YOUR_AVION_HOME_PATH>/third_party/decord/build/libdecord.so)``` in step (3), it means that you have mismatched GLIBCXX versions between the ubuntu system and anaconda environment. As a workaround, you can simply copy the system libstdc++ to your anaconda libstdc++ and update the soft link, for example (the version number might vary due to your system) 

```bash
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 <YOUR_CONDA_HOME_PATH>/envs/avion/lib/
rm  <YOUR_CONDA_HOME_PATH>/envs/avion/lib/libdecord.so
ln -s <YOUR_CONDA_HOME_PATH>/envs/avion/lib/libdecord.so.6.0.30 <YOUR_CONDA_HOME_PATH>/envs/avion/lib/libdecord.so
```

* If you see an incorrect path, e.g. `<YOUR_CONDA_HOME_PATH>/envs/avion/lib/python3.10/site-packages/decord`, it means that you are using a wrong decord which may have previously been installed using `pip install decord`. You can `pip uninstall decord` and re-install following the steps above.
ry due to 
* If you see errors not listed, please create an issue.



## Preparing datasets
If you want to train/evaluate on the benchmark, please refer to [datasets/README.md](../datasets/README.md) to see how we prepare datasets for this project.
