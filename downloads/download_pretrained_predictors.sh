#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/predictor_weights.zip -O $parentdir/../data/tmp/predictor_weights.zip
unzip $parentdir/../data/tmp/predictor_weights.zip -d $parentdir/../data
