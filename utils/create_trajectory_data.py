from MOTSynth import MOTSynth
import pandas as pd
import warnings
import os
warnings.simplefilter(action='ignore')

import argparse

parser = argparse.ArgumentParser(description='Process sequences')
parser.add_argument('--sequence',type=int)

args = parser.parse_args()


sequence = args.sequence

mot_synth = MOTSynth()

if int(mot_synth.isMoving(sequence)) == 0:
    mot_synth.loadMerged(sequence)
    
    gt = mot_synth.merged[pd.isnull(mot_synth.merged.id_x) == False]
    

    step = 12

    filtered_gt = gt[gt.frame % step == 0 ]

    filtered_gt.sort_values(["id_x", "frame"], inplace = True)
    id_split = filtered_gt.groupby("id_x")["frame"].diff(-1).reset_index() 

    id_split["frame"].fillna(-12, inplace = True)

    filtered_gt["dt"] = id_split["frame"].tolist()
    final_data_xy = pd.DataFrame(columns = ["frame", "id", "x", "y"])
    final_data_uv = pd.DataFrame(columns = ["frame", "id", "x", "y"])
    old_id = -1

    id = -1 
    filtered_gt["bb_center"] = filtered_gt["bb_left_x"] + (filtered_gt["bb_width_x"]) /2. 
    filtered_gt["bb_bottom"] = (filtered_gt["bb_height_x"] + filtered_gt["bb_top_x"])
    for index, row in filtered_gt.iterrows():
        
        if not old_id == row["id_x"]:
            
            old_id = row["id_x"]
            id+=1
        if not row["dt"] == -12:
            id+=1 
        
        final_data_xy.loc[len(final_data_xy)] = [int(row["frame"]//12), int(id), row["x"], row["y'"]]
        final_data_uv.loc[len(final_data_uv)] = [int(row["frame"]//12), int(id), row["bb_center"], row["bb_bottom"]]


    final_data_xy["frame"] = final_data_xy["frame"].astype("int")
    final_data_xy["id"] = final_data_xy["id"].astype("int")

    final_data_uv["frame"] = final_data_uv["frame"].astype("int")
    final_data_uv["id"] = final_data_uv["id"].astype("int")
    print(final_data_uv)
    print(final_data_xy)
    print(mot_synth.root_dir)
    final_data_uv.to_csv(
        os.path.join(mot_synth.root_dir, "dataset", "uv", "{:03d}.txt".format(sequence)), 
        header=False, index=False, sep='\t')
    final_data_xy.to_csv(
        os.path.join(mot_synth.root_dir, "dataset","xy", "{:03d}.txt".format(sequence)), 
        header=False, index=False, sep='\t')