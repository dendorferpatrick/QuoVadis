
import argparse
from quovadis.bev_reconstruction import run_homography
parser = argparse.ArgumentParser()
# basic experiment setting
parser.add_argument('--dataset', default="MOT17",
                    type=str, help="Dataset")
parser.add_argument('--sequence', type=str,
                    default="MOT17-02", help="sequence")
args = parser.parse_args()
run_homography(args.dataset, args.sequence)
