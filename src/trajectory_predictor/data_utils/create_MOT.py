import shutil

import json
import os
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
DATAPATH = "/storage/user/dendorfp/MOT16"
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/utils")
from loader import PanopticLoader

panoptic_loader = PanopticLoader()

def compute_splits(split = "test"):
    seqs = ["MOT16-02"]
    return seqs

def create_data(split):

    split_folder = os.path.join("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/Predictors/data/datasets/mot16", split)
    os.makedirs(split_folder, exist_ok = True)
    seqs = compute_splits(split)
    # seqs = ["715"]

    for seq in tqdm(seqs):

        img_rgb = Image.open(os.path.join(DATAPATH, "mapping", seq, "rgb_interpolate.png" )).convert("RGB")
        
       
        img_rgb.save(os.path.join(DATAPATH, "mapping", seq, "trajectory_rgb_interpolate.png" ))
        img_rgb.save(os.path.join(split_folder, "{}.png".format(seq)))

       
        img_classes = Image.open(os.path.join(DATAPATH, "mapping", seq, "classes_interpolate.png" ))

        img_classes =np.uint(np.array(img_classes)/255. * 8 )
        h, w = img_classes.shape
        new_img = np.zeros((h, w, 3))
        for i in np.unique(img_classes):
            new_img[img_classes == i] = np.array(panoptic_loader.id_colors[i])
        
        im_class = Image.fromarray(new_img.astype(np.uint8))
        im_class.save(os.path.join(DATAPATH, "mapping", seq, "trajectory_classes_interpolate.png" ))
        im_class.save( os.path.join(split_folder, "{}-op.png".format(seq)))
                     



if __name__ == "__main__":
    for split in ["train"]:
        print(split)
        create_data(split)
