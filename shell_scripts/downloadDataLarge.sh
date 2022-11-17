#!/bin/bash

mkdir data/acdc_timelapse_full
mkdir data/acdc_timelapse_full/test
mkdir data/acdc_timelapse_full/train
mkdir data/acdc_timelapse_full/test/images
mkdir data/acdc_timelapse_full/test/annotations
mkdir data/acdc_timelapse_full/train/images
mkdir data/acdc_timelapse_full/train/annotations
mkdir data/raw_data

cd data/raw_data

wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/x3eg2iiaLJnbzCY/download/acdc_data2.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/drXrEjZpYQN4KZT/download/acdc_data3.zip

if ! command -v unzip /dev/null
then
    sudo apt-get install unzip -y
fi
unzip acdc_data2.zip -d acdc_data2
rm acdc_data2.zip

unzip acdc_data3.zip -d acdc_data3
rm acdc_data3.zip

cd ../..
