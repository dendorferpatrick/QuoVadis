import pandas as pd
import numpy as np 
import os
import sys
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/datasets")
from MOT import MOTTracking

data_path = "Predictors/data/datasets"
def get_y0(H, img_width):
        
        x_array = np.arange(0, img_width)
        horizon = -(H[2, 0] * x_array + H[2, 2] )/ H[2, 1]
        y0_list = []
        for h, x in zip(horizon, x_array):

            y = np.arange( np.ceil(h)+1, 1080)

            xx = np.ones(len(y)) * x
            p = np.stack((xx, y, np.ones(len(y))))
            pp = H.dot(p).T
            pp = pp[:, :2]/pp[:, -1:]
            dd = pp[1:, 1] - pp[:-1, 1]

            dk = dd[1:]/ dd[:-1]

            pix_y = y[1:]
            lower_threshold = pix_y[abs(dd) > .2]

            if len(lower_threshold) == 0 :
                y0_list.append(h + 100)
            else: y0_list.append(lower_threshold[-1])
        return np.array(y0_list)

def pix2real(H, pos,pixels, y0, img_width):
    x_pix = np.clip(pixels[:, 0], 0, img_width-1).astype(int)

    Ay = (H[1, 0] * pixels[:, 0] +  H[1, 1] * y0[x_pix] + H[1,2])
    Ax = (H[0, 0] * pixels[:,0] + H[0, 1] *  y0[x_pix] + H[0,2])
    B = (( H[2, 0]*pixels[:, 0] +   H[2, 1] * y0[x_pix] + H[2,2]))


    mask = pixels[:, 1] < y0[x_pix]
    converted_y =  (Ay/B - Ay/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix])) 
    converted_y[np.isnan(converted_y)] = 0

    converted_x = (Ax/B - Ax/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix]))
    converted_x[np.isnan(converted_x)] = 0
    pos[:,1 ] = pos[:, 1] * (1-mask)  + converted_y * mask
    pos[:,0 ] = pos[:, 0] * (1-mask)  + converted_x * mask

    return pos



def create_data(challenge, sequence_nr ):
    mot = MOTTracking(partition = ["{:02d}".format(sequence_nr)], 
                                    challenge = challenge, 
                                    fields=["rgb",
                                    "homography_depth", 
                                    "labels",     
                                    
                                    ])
    seq  = mot.data.sequences[0]
    item = seq.__getitem__(1, ["rgb", "homography_depth"])
    height, width, _ = item["rgb"].shape
    labels = seq.labels
    labels[labels["class"] == 1]
    H = np.array(item["homography_depth"]["IPM"])
    

    labels["u"] = labels["bb_left"] + 1/2.*labels["bb_width"]
    labels["v"] = labels["bb_top"] + labels["bb_height"]

    pixel_positions = labels[["u", "v"]].values
    pixel_positions = np.concatenate((pixel_positions, np.ones((len(pixel_positions), 1)) ),1)
    

        
    
    y0 = get_y0(H, img_width = width)
    

    pos = H.dot(pixel_positions.T).T
    pos = pos[:, :-1]/pos[:, -1:]

    if y0 is not None:

        new_pos =  pix2real(H, pos*1.,pixel_positions*1., y0, width )
    else:
        new_pos = pos*1.
    
    labels[["x", "y"]] = new_pos
    labels = labels[["frame","id","x","y","u","v","bb_left","bb_top","bb_width","bb_height"]]
    if challenge == "MOT20":
        save_path = os.path.join(data_path, "mot20", "test", f"test_MOT20_{sequence_nr:02d}.txt")
    elif challenge == "MOT16":
        save_path = os.path.join(data_path, "mot17", "test", f"test_MOT17_{sequence_nr:02d}.txt")
    labels.to_csv(save_path, index = False)
    
data  = [["MOT20", 1], 
["MOT20", 2], 
["MOT20", 3], 
["MOT16", 2], 
["MOT16", 4], 
["MOT16", 9]]

for challenge , sequence_nr  in data: 
    create_data(challenge, sequence_nr)