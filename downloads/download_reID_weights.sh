#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/resnet50_market_xent.pth.tar -O $parentdir/../data/tmp/resnet50_market_xent.pth.tar
mkdir -p $parentdir/../data/reID_weights
cp -r  $parentdir/../data/tmp/resnet50_market_xent.pth.tar $parentdir/../data/reID_weights

