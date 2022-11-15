#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT20_tracker.zip -O $parentdir/../data/tmp/MOT20_tracker.zip
unzip $parentdir/../data/tmp/MOT20_tracker.zip -d $parentdir/../data