#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/../create_data_folder.sh

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT20_features.zip -O $parentdir/../../data/tmp/MOT20_features.zip
unzip -o $parentdir/../../data/tmp/MOT20_features.zip -d $parentdir/../../data