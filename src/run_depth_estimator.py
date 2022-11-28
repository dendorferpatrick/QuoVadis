import argparse
from adabins import infer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script for AdaBins')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model', required=True)
    parser.add_argument('--input_path', type=str,
                        help='Path to input images', required=True)
    parser.add_argument('--output_path', type=str,
                        help='Path to save predictions', required=True)
    parser.add_argument('--images', nargs='+', default=[],
                        help='List of individuals images in <images_dir>', required=False)
    parser.add_argument('--save-as-png', action="store_true", default=False,
                        help='Save depth image also as png', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    infer.infer_depth(args.model_path, args.input_path,
                      args.output_path, args.images, args.save_as_png)
