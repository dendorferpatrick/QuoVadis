import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--challenge",
        default="MOT17",
        type=str
    )
  

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Baseline tracker",
    )


  
    return parser



args = get_parser().parse_args()

image_folder = "/storage/user/dendorfp/MOT/{MOT}/{split}".format(split = args.split, MOT = args.challenge)

sequences = os.listdir(image_folder)

print(sequences)
for seq in sequences:
    

    command = "utils/panoptic?segmentation/extract_panoptic_segemntation.py --image_folder {image_folder}/{sequence}/img1 --output_folder /storage/user/dendorfp/MOT/{MOT}/{split}/{sequence} --sequence {sequence}".format(sequence= seq, 
    MOT = args.challenge, split = args.split, image_folder =  image_folder)
    os.system("sbatch run_slurm.sbatch {}".format(command))
