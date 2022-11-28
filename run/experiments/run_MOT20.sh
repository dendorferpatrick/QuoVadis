#!/bin/sh
TRACKERS=('CenterTrack' 'ByteTrack')
for tracker in "${TRACKERS[@]}"; do
    python src/run_quovadis.py --config-file ./run/cfgs/MOT20.yaml --dataset MOT20 --tracker-name $tracker --sequences MOT20-01 MOT20-02 MOT20-03 MOT20-05 --save --eval
done
