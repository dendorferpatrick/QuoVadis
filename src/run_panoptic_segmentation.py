import argparse
import glob

from quovadis.bev_reconstruction import run_panoptic_segmentation


def get_parser():
    parser = argparse.ArgumentParser(
        description="Running detectron2 panoptic segmentation on images")
    parser.add_argument(
        "--dataset",
        default="MOT17",
        type=str
    )

    parser.add_argument(
        "--sequences",
        nargs='+',
        default=["MOT17-02"],
        help="Names of sequence for panoptic segmentation",
    )
    parser.add_argument('--single', default=False,
                        action="store_true", help="only process first frame")
    return parser


args = get_parser().parse_args()

for seq in args.sequences:

    image_folder = f'./data/{args.dataset}/sequences/{seq}/img1'
    output_folder = f'./data/{args.dataset}/sequences/{seq} --sequence {seq}'

    frames = sorted(glob.glob("{}/*.png".format(image_folder)) +
                    glob.glob("{}/*.jpg".format(image_folder)))
    if args.single:
        frames = frames[:1]

    run_panoptic_segmentation(frames, output_folder, seq)
