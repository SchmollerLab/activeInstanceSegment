#!/bin/bash

mkdir data/raw_data/acdc_large

cd data/raw_data/acdc_large

wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/x3eg2iiaLJnbzCY/download/acdc_data2.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/drXrEjZpYQN4KZT/download/acdc_data3.zip

if ! command -v 7z /dev/null
then
    sudo apt install p7zip-full -y
fi
7z x acdc_data2.zip
rm acdc_data2.zip

7z x acdc_data3.zip
rm acdc_data3.zip

cd ../../..
