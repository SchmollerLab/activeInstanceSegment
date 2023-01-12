#!/bin/bash

mkdir $DATA_PATH/raw_data/acdc_large

cd $DATA_PATH/acdc_large
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/L57ZkKqKz8QcDoE/download/TimeLapse_2D.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/x3eg2iiaLJnbzCY/download/acdc_data2.zip
wget https://hmgubox2.helmholtz-muenchen.de/index.php/s/drXrEjZpYQN4KZT/download/acdc_data3.zip

7z x TimeLapse_2D.zip
7z x acdc_data2.zip
7z x acdc_data3.zip

cd $PROJECT_ROOT
