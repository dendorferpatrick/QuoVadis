#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh

wget -nc https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/quickstart_tracker.zip -O $parentdir/../data/tmp/quickstart_tracker.zip
unzip -o $parentdir/../data/tmp/quickstart_tracker.zip -d $parentdir/../data

wget -nc https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/quickstart_features.zip -O $parentdir/../data/tmp/quickstart_features.zip
unzip -o $parentdir/../data/tmp/quickstart_features.zip -d $parentdir/../data

wget -nc https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/quickstart_data.zip -O $parentdir/../data/tmp/quickstart_data.zip
unzip -o $parentdir/../data/tmp/quickstart_data.zip -d $parentdir/../data

echo "Download pretrained predictor weights"
bash $parentdir/download_pretrained_predictors.sh