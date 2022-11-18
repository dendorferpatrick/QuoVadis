#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"

wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/AdaBins_synthetic.pt -O $parentdir/../data/tmp/AdaBins_synthetic.pt
mkdir -p $parentdir/../data/AdaBins_weights
cp -r  $parentdir/../data/tmp/AdaBins_synthetic.pt $parentdir/../data/AdaBins_weights



