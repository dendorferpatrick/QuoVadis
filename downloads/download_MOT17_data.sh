#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh


wget https://motchallenge.net/data/MOT17.zip -O $parentdir/../data/tmp/MOT17.zip
unzip -o $parentdir/../data/tmp/MOT17.zip -d $parentdir/../data/tmp/tmpMOT17

mkdir -p $parentdir/../data/MOT17/sequences
mv $parentdir/../data/tmp/tmpMOT17/MOT17/train/*FRCNN  $parentdir/../data/MOT17/sequences/
mv $parentdir/../data/tmp/tmpMOT17/MOT17/test/*FRCNN  $parentdir/../data/MOT17/sequences/

CURRENTDIR="$(pwd)"
cd $parentdir/../data/MOT17/sequences/
for seq in *
do

test -d $seq || continue
echo $seq | grep -q FRCNN || continue
mv $seq $(echo $seq | sed -e 's/-FRCNN//')
done


cd $CURRENTDIR
