import shutil

import json
import os
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image
DATAPATH = "/storage/user/dendorfp/MOTSynth"
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/utils")
from loader import PanopticLoader

panoptic_loader = PanopticLoader()

def compute_splits(split = "train"):
    train = 0.45
    val = 0.15
    assert (train + val) < 1
    test = 1- val - train

    seqs = os.listdir(os.path.join(DATAPATH, "frames"))
    exclude = ["629", "757"]
    for s in exclude:

        seqs.remove(s)

    print(len(seqs))

    import random
    random.Random(4).shuffle(seqs)
    print(seqs)
    train_index_start = 0 
    train_index_end = val_index_start = int(len(seqs) * train)
    val_index_end =  test_index_start = int(len(seqs) * (train + val))
    test_index_end = len(seqs)

    train_data = seqs[train_index_start: train_index_end]
    val_data = seqs[val_index_start: val_index_end]
    test_data = seqs[test_index_start: test_index_end]




    print(len(train_data) + len(val_data) + len(test_data))

    if split == "train": 
        return train_data
    elif split == "val":
        return val_data
    elif split == "test":
        return test_data
    else: 
        return False


def create_data(split):

    split_folder = os.path.join("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/Predictors/data/datasets/motsynth", split)
    os.makedirs(split_folder, exist_ok = True)
    seqs = compute_splits(split)
    # seqs = ["715"]

    for seq in tqdm(seqs):

        # load annotations

        df = pd.read_csv(os.path.join(DATAPATH, "mot_annotations", seq, "gt", "gt.txt"))

        # load map metadata
        metadata_json = os.path.join(DATAPATH,"mapping", seq, "mapping.json")
        print(df.head())
        with open(metadata_json, 'r') as fp:
            metadata = json.load(fp)
        print(metadata)
        df["x_world"] = df["x_world"] - metadata["x_min"]
        df["y_world"] = df["y_world"] - metadata["y_min"]
        print(df.head())
        trajectory_data = df[["frame", "id", "x_world", "y_world"]]
        trajectory_data.rename(columns = {"x_world": "x", "y_world": "y"}, inplace = True)


        trajectory_data.to_csv(os.path.join(split_folder, "{}_{}.txt".format(split, seq)), index = False)
        
        
        img_rgb = Image.open(os.path.join(DATAPATH, "mapping", seq, "rgb_interpolate.png" )).convert("RGB")
        
        # img_rgb = np.array(img_rgb)[:, :, :3]

        # img_rgb = Image.fromarray(img_rgb.astype(np.uint8))
        img_rgb.save(os.path.join(DATAPATH, "mapping", seq, "trajectory_rgb_interpolate.png" ))
        img_rgb.save(os.path.join(split_folder, "{}.png".format(seq)))

        # shutil.copyfile(os.path.join(DATAPATH, "mapping", seq, "rgb_interpolate.png" ), os.path.join(split_folder, "{}.png".format(seq)))
        # shutil.copyfile(os.path.join(DATAPATH, "mapping", seq, "classes_interpolate.png" ), os.path.join(split_folder, "{}-op.png".format(seq)))
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
    for split in ["train", "val", "test"]:
        print(split)
        create_data(split)
