#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

# Download tracking data
echo "Downloading tracking data MOT17 & MOT20"
bash $parentdir/download_MOT17_tracker.sh
bash $parentdir/download_MOT20_tracker.sh

echo "Downloadings feature data MOT17 & MOT20"
bash $parentdir/download_MOT17_visual_features.sh
bash $parentdir/download_MOT20_visual_features.sh

echo "Download pretrained predictor weights"
bash $parentdir/download_pretrained_predictors.sh



