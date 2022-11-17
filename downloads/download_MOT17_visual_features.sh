#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT17_features.zip -O $parentdir/../data/tmp/MOT17_features.zip
unzip -o $parentdir/../data/tmp/MOT17_features.zip -d $parentdir/../data