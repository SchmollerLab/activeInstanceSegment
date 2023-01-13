#! /bin/bash

cd /mnt

sudo apt-get update -y
#sudo apt-get upgrade -y


echo install python and pip
sudo apt install python3-pip -y
sudo apt-get install python-is-python3 -y


echo installing cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get -y install cuda

cd activeCell-ACDC

echo installing pytorch
pip3 install torch torchvision torchaudio

echo installing detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip3 install -r ./requirements.txt


sudo apt install p7zip-full -y
sudo apt install libgl1-mesa-glx -y


echo "set enviromental variables"

project_root="$(pwd)"/activeCell-ACDC
data_path="$(pwd)"/activeCell-ACDC/data
mkdir ./data
mkdir ./data/raw_data
mkdir ./output

if [[ -z $IS_SERVER ]]; then
    echo "enviromental IS_SERVER is not set. appending to .bashrc"
    echo IS_SERVER=true | sudo tee -a /etc/environment
else
    echo "enviromental IS_SERVER is already set."
fi

if [[ -z $PROJECT_ROOT ]]; then
    echo "enviromental PROJECT_ROOT is not set. appending to .bashrc"
    echo PROJECT_ROOT=\"$project_root\" | sudo tee -a /etc/environment
else
    echo "enviromental PROJECT_ROOT is already set."
fi

if [[ -z $DATA_PATH ]]; then
    echo "enviromental DATA_PATH is not set. appending to .bashrc"
    echo DATA_PATH=\"$data_path\" | sudo tee -a /etc/environment
else
    echo "enviromental DATA_PATH is already set."
fi

echo setting up data
./shell_scripts/downloadDataLarge.sh
python src/dataloader/data2coco.py