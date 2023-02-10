#!/bin/bash

mkdir ./data
mkdir ./data/raw_data
mkdir ./data/raw_data/acdc_large_cls

cd ./data/raw_data/acdc_large_cls
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/L57ZkKqKz8QcDoE/download/TimeLapse_2D.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/x3eg2iiaLJnbzCY/download/acdc_data2.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/drXrEjZpYQN4KZT/download/acdc_data3.zip

7z x TimeLapse_2D.zip
7z x acdc_data2.zip
7z x acdc_data3.zip

cd ../../..
