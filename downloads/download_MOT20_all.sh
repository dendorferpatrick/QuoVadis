#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

# Download tracking data
echo "Downloading tracking data  MOT20"
bash $parentdir/download_MOT20_tracker.sh

echo "Downloadings feature data  MOT20"
bash $parentdir/download_MOT20_visual_features.sh

echo "Download pretrained predictor weights"
bash $parentdir/download_pretrained_predictors.sh



