#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

# Download tracking data
echo "Downloading tracking data MOT20 "
bash $parentdir/MOT20/download_MOT20_tracker.sh

echo "Downloading features data MOT20"
bash $parentdir/MOT20/download_MOT20_visual_features.sh

echo "Downloading map info MOT20"
bash $parentdir/MOT20/download_MOT20_maps.sh

echo "Downloading calibration info MOT20"
bash $parentdir/MOT20/download_MOT20_calib.sh

echo "Downloading homographies for MOT20"
bash $parentdir/MOT20/download_MOT20_homographies.sh

echo "Download pretrained predictor weights"
bash $parentdir/download_pretrained_predictors.sh



