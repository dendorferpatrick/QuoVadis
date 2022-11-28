#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh


wget https://motchallenge.net/data/MOT20.zip -O $parentdir/../data/tmp/MOT20.zip
unzip -o $parentdir/../data/tmp/MOT20.zip -d $parentdir/../data/tmp/tmpMOT20

mkdir -p $parentdir/../data/MOT20/sequences
mv $parentdir/../data/tmp/tmpMOT20/MOT20/train/*  $parentdir/../data/MOT20/sequences/
mv $parentdir/../data/tmp/tmpMOT20/MOT20/test/* $parentdir/../data/MOT20/sequences/
