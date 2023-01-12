#! /bin/bash

sudo apt-get update
sudo apt-get upgrade -y


echo install python and pip
sudo apt install python3-pip
sudo apt-get install python-is-python3


echo installing cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

echo installing pytorch
pip3 install torch torchvision torchaudio

echo installing detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip3 install -r ./requirements.txt


