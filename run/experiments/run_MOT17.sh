#!/bin/sh
TRACKERS=('ByteTrack' 'CenterTrack' 'qdtrack' 'CSTrack' 'FairMOT' 'JDE' 'TraDeS' 'TransTrack')
for tracker in "${TRACKERS[@]}"; do
    python src/run_quovadis.py --config-file ./run/cfgs/MOT17.yaml --dataset MOT17 --tracker-name $tracker --sequences MOT17-02 MOT17-04 MOT17-09 --save --eval
done