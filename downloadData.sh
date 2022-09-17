#!/bin/bash

mkdir data
mkdir data/dataInCOCO
mkdir data/dataInCOCO/images
mkdir data/dataInCOCO/annotations

cd data
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/DdXYAam2mRwZn88/download/TimeLapse_2D.zip

if ! command -v unzip /dev/null
then
    sudo apt-get install unzip - y
fi
unzip TimeLapse_2D.zip -d TimeLapse_2D
rm TimeLapse_2D.zip
cd ..
