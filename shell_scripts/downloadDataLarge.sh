#!/bin/bash

mkdir data/raw_data
mkdir data/raw_data/acdc_large

cd data/raw_data/acdc_large

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

cd ../../..
