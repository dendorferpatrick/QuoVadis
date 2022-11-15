
import sys
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/datasets")

from MOTSynth import MOTSynthTracking
import os
import pandas as pd
import multiprocessing
import json
import itertools
import numpy as np
def compute_ratio_track(sequence_nr):
    motsynth = MOTSynthTracking(partition = ["{}".format(sequence_nr)], 
                                    fields = ["rgb", "homography_gt"])
    position_file = "/storage/user/dendorfp/MOTSynth/positions_h_gt/{}.txt".format(sequence_nr)
    position_df = pd.read_csv(position_file)


    pos_pixels = position_df[['u', 'v']].values

    pos_pixels = pos_pixels.astype(int)
    pos_pixels = pos_pixels[:, (1, 0)]
    pos_pixels[:, 0]  = np.clip(pos_pixels[:, 0], 0, 1079)
    pos_pixels[:, 1]  = np.clip(pos_pixels[:, 1], 0, 1919)
    
 
 
    seq = motsynth.data.sequences[0]
    frame = 1 
    item = seq.__getitem__( frame)
    
    img = item["rgb"]
    
    mask_shape = img.shape
    pixels = np.array(list(itertools.product(range(mask_shape[0]+2), range(mask_shape[1]+2))))
    pixels = pixels[:, (1, 0)]
    IPM = np.array(item["homography_gt"]["IPM"])

    p_c = np.concatenate((pixels, np.ones((len(pixels), 1))), 1)
    world_coordinates = IPM.dot(p_c.T).T
    world_coordinates = world_coordinates[:, :2]/world_coordinates[:, -1:]

    world_image =np.reshape(world_coordinates, ( mask_shape[0] + 2, mask_shape[1] + 2, 2 ))
    dx = world_image[:-1, 1:] - world_image[:-1, :-1]
    dy = world_image[1:,:-1] - world_image[:-1, :-1]
    length_dx = np.sqrt(np.sum(dx**2, -1))
    length_dy = np.sqrt(np.sum(dy**2, -1))
    length_ratio = length_dx/ length_dy
    
    dx_area = abs(dx[..., 0]* dx[..., 0] + dx[..., 1]* dx[..., 1])
    area = abs(dy[..., 0]* dx[..., 1] - dy[..., 1]* dx[..., 0])
    value_abs = dx_area[1079, int(1920 /2)]/ area
    values_abs = value_abs[pos_pixels[:,0], pos_pixels[:, 1]]
    position_df[["ratio_abs"]] = values_abs

    value_rel = dx_area/ area
    values_rel = value_rel[pos_pixels[:,0], pos_pixels[:, 1]]
    position_df[["ratio_rel"]] = values_rel
    position_df[["dist_rel"]] = length_ratio[pos_pixels[:,0], pos_pixels[:, 1]]

    values_area =  area[pos_pixels[:,0], pos_pixels[:, 1]]
    position_df[["area"]] = values_area
    position_df[["area_rel"]] = values_area / area[pos_pixels[:,0] + 1, pos_pixels[:, 1] + 1]
    position_df = position_df[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "x_pixel", "y_pixel",
     "x", "y", "z", "u", "v", "H_x", "H_y", "ratio_abs",  "ratio_rel", "area", "dist_rel", "area_rel"]]
    position_df.to_csv(position_file, index = False)




if __name__ == "__main__": 
   

    data_folder = "/storage/user/dendorfp/MOTSynth/mot_annotations"
    sequences = os.listdir(data_folder)
    seq_df = pd.read_csv("/storage/user/dendorfp/MOTSynth/statistics/sequences_list.txt", sep=";")

    # filter only static scenes 
    seq_df = seq_df[seq_df.moving == 0]
    
    with open('/usr/wiss/dendorfp/dvl/projects/TrackingMOT/splits/MOTSynth/test.json', 'r') as f:
        data = json.load(f)

    

    # # df_stats = pd.read_csv("/storage/user/dendorfp/MOTSynth/statistics/stats.txt", names = ["sequence", "H_L", "H_T", "H_E"])
    # print(data)
    seq_allowed = [int(name.split("-")[-1]) for name in seq_df.name if int(name.split("-")[-1]) in data]
    # print(len(seq_allowed), len(seq_df), len(data))
    # # required_sequences = set(seq_allowed).difference(set(df_stats.sequence))
    
    pool = multiprocessing.Pool(processes=20)
    
    pool.map(compute_ratio_track, ["{:03d}".format(seq) for seq in seq_allowed])