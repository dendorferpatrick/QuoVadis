import sys 
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/datasets")
from MOT import MOTTracking
from PIL import Image
import json
import argparse
import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
import multiprocessing
import traceback

DATAFOLDER = "/storage/user/dendorfp/MOT20/positions_gt"
def create_position(sequence):
    try:
        mot = MOTTracking(partition = ["{:02d}".format(sequence)], challenge="MOT20",  fields = ["labels",  "homography_depth"])


        seq = mot.data.sequences[0]
        print(seq)
        print(seq.labels)

        labels = seq.labels
        item = seq.__getitem__(1, "homography_depth")
        H = item["homography_depth"]
        H = H["IPM"]
        print(H, labels)

        labels["u"] = labels["bb_left"] + labels["bb_width"]/2
        labels["v"] = labels["bb_top"] + labels["bb_height"]
        uv = labels[["u", "v"]].values
        vector_mat = np.concatenate(
            (uv, np.ones((len(uv), 1))), axis=-1)
        trans = np.dot(H, vector_mat.T)


        trans = trans/trans[-1, :]
        xy = trans[:2].T
        labels[["H_x", "H_y"]] = xy

      
        labels[["frame", "id"]] = labels[["frame", "id"]].astype("int")
        labels = labels[(labels["class"] == 1)]
        labels.to_csv(os.path.join(DATAFOLDER, "MOT20-{:02d}.txt".format(sequence)), index = False)

        labels["x"] = labels["H_x"]
        labels["y"] = labels["H_y"]
        labels[["frame", "id", "x", "y", "u", "v", "bb_left", "bb_top", "bb_width", "bb_height"]].to_csv(
            os.path.join("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/Predictors/data/datasets/mot20/test", "test_MOT20-{:02d}.txt".format(sequence)), index = False)




    except:
        print(traceback.print_exc())
        pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sequence', required = False, type=int, 
    #                             help='challenge sequence')
    # parser.add_argument('--start', required = True, type=int, 

    #                             help='challenge sequence')
    # parser.add_argument('--end', required = True, type=int, 
    #                             help='challenge sequence')
                                
    # args = parser.parse_args() 

    
    
    pool = multiprocessing.Pool(processes=4)
   

    # for sequence in np.arange(1):
    #     # args = parser.parse_args()
    #     create_position(sequence)


    sequences =  [1, 2, 3, 5]  

    pool.map(create_position, sequences)