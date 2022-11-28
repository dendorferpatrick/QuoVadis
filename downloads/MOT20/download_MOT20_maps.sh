#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/../create_data_folder.sh

wget  https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT20_maps.zip -O $parentdir/../../data/tmp/MOT20_maps.zip
unzip -o $parentdir/../../data/tmp/MOT20_maps.zip -d $parentdir/../../data
