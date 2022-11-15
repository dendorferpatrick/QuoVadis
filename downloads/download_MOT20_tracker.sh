#!/bin/sh
parentdir="$(dirname "$(realpath $0)")"
bash $parentdir/create_data_folder.sh

<<<<<<< HEAD
wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT20_tracker.zip -O $parentdir/../data/tmp/MOT20_tracker.zip
unzip $parentdir/../data/tmp/MOT20_tracker.zip -d $parentdir/../data
=======
wget https://vision.in.tum.de/webshare/g/papers/dendorfer/quovadis/MOT20_tracker.zip
unzip MOT20_tracker.zip
rm MOT20_tracker.zip
>>>>>>> 75a783e10d3c796f7f660a85f5a544c1dbd546a1
