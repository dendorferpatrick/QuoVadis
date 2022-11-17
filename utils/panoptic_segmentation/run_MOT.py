import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Running detectron2 panoptic segmentation on images")
    parser.add_argument(
        "--challenge",
        default="MOT17",
        type=str
    )
  

    parser.add_argument(
        "--sequences",
        nargs='+',
        default=["MOT17-02"],
        help="Names of sequence for panoptic segmentation",
    )

    return parser



args = get_parser().parse_args()

image_folder = "./data/{MOT}".format(MOT = args.challenge)
for seq in args.sequences:
    
    print(seq)
    command = f'./utils/panoptic_segmentation/extract_panoptic_segmentation.py --single --image_folder {image_folder}/sequences/{seq}/img1 --output_folder {image_folder}/sequences/{seq} --sequence {seq}'
    os.system(f'python {command}')
