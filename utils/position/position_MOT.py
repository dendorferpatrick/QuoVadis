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


def extract(frame, mot):
        columns = ["frame", "id", "x", "y" , "z", "height"]
        for quantile in np.arange(0.1, 0.99, 0.1): 
            columns.append("{}_quantile".format(quantile))
        df = pd.DataFrame(columns =["frame", "id", "x", "y" , "z", "height"])
    
        gradient_threshold = 30
        gradient_mask = mot.data.sequences[0].get_depth_gradient_mask(frame, gradient_threshold )
        
        pedestrian_pixels = mot.data.sequences[0].get_pedestrian_pixels(frame, gradient_mask)
        
        points, colors, _, img, pt_cloud = mot.data.sequences[0].transform_lidar_world_mot(frame)
        
        ped_positions = mot.data.sequences[0].get_pedestrian_positions_mot(pedestrian_pixels, points)
    

            

        for id, position in ped_positions.items():
            pos = position["position"]
            x = pos[0]
            y = pos[2]
            z = pos[1] 
            height = position["height"]
            row = [frame, id, x, y, z, height]
            for i in range(3, len(pos)):
                row.append(pos[i])
            df.loc[len(df)] = [frame, id, x, y, z, height]
        

        return df
DATAFOLDER = "/storage/user/dendorfp/MOT16/positions"
def create_position(sequence):
    try:
        mot = MOTTracking(partition = ["{:02d}".format(sequence)], fields = ["rgb","dets",  "depth", "segmentation", 
                                                              "lidar", "pose", "calibration"])
        

        res = [extract(frame,mot ) for frame in tqdm(np.arange(1, mot.data.sequences[0].__len__() + 1))]
    
        df = pd.concat(res)
    
        df[["frame", "id"]] = df[["frame", "id"]].astype("int")
        
        df.to_csv(os.path.join(DATAFOLDER, "MOT16-{:02d}.txt".format(sequence)), index = False)
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


    sequences =  [2, 4, 9] 

    pool.map(create_position, sequences)