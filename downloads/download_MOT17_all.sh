#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

# Download tracking data
echo "Downloading tracking data MOT17 "
bash $parentdir/MOT17/download_MOT17_tracker.sh

echo "Downloading features data MOT17"
bash $parentdir/MOT17/download_MOT17_visual_features.sh

echo "Downloading map info MOT17"
bash $parentdir/MOT17/download_MOT17_maps.sh

echo "Downloading calibration info MOT17"
bash $parentdir/MOT17/download_MOT17_calib.sh

echo "Downloading homographies for MOT17"
bash $parentdir/MOT17/download_MOT17_homographies.sh

echo "Download pretrained predictor weights"
bash $parentdir/download_pretrained_predictors.sh



