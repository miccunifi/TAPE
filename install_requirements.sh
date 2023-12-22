#!/bin/bash

# Install packages
conda install -y pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas==2.1.3 matplotlib==3.8.2 pyyaml==6.0.1 dotmap==1.3.30 tqdm==4.66.1 comet-ml==3.35.3 git+https://github.com/openai/clip.git@a1d0717 scikit-image==0.22.0 opencv-python==4.8.1.78 einops==0.7.0 lmdb==1.4.1 pytorch-lightning==2.1.2 lpips==0.1.4 matplotlib==3.8.2