from kalman_filter import KalmanSmoother
from tqdm import tqdm
import warnings
from MOTSynth import MOTSynth
import numpy as np
import os
warnings.simplefilter(action='ignore')

import argparse

parser = argparse.ArgumentParser(description='Process sequences')
parser.add_argument('--sequence',type=int)

args = parser.parse_args()


sequence = args.sequence


motsynth = MOTSynth()
if int(motsynth.isMoving(sequence)) == 0:
    motsynth = MOTSynth()
    motsynth.loadMerged(sequence) 

    motsynth.merged.sort_values("frame", inplace= True)

    motsynth.merged["bb_center"] = motsynth.merged["bb_left_x"] + (motsynth.merged["bb_width_x"]) /2. 
    motsynth.merged["bb_bottom"] = (motsynth.merged["bb_height_x"] + motsynth.merged["bb_top_x"])
    detection_df = motsynth.merged[motsynth.merged.detection == 1.0]
    
    frames_diff = detection_df.groupby(["matched_id"])["frame"].diff(-1)
    detection_df["dt"] = frames_diff
    motsynth.merged[["smooth_x", "smooth_y", "pred_x", "pred_y","smooth_u", "smooth_v", "pred_u", "pred_v",  "dt", "gap"]] = None

    unique_ids = detection_df.matched_id.unique()
    print("unique_ids", unique_ids)
    for id in detection_df.matched_id.unique():

        df_id = detection_df[detection_df.matched_id == id]
        if len(df_id) == 0: 
            continue
        trajectory_bb = df_id[["bb_center", "bb_bottom"]].values[0]
        trajectory_xy = df_id[["x", "y'"]].values[0]

        KF_xy = KalmanSmoother(  np.array([trajectory_xy[0] , trajectory_xy[1], 0.0, 0.]))
        KF_uv = KalmanSmoother(np.array([trajectory_bb[0] , trajectory_bb[1], 0.0, 0.]))

        for index, row in (df_id.iterrows()):
        
            dt = row["dt"]
            if dt == -1:
                
                xy = KF_xy.step(row[["x", "y'"]].values, decrease_process_uncertainty=True)
                uv = KF_uv.step(row[["bb_center", "bb_bottom"]].values, decrease_process_uncertainty=True)
                motsynth.merged.loc[(( motsynth.merged.frame == row["frame"]) & (motsynth.merged.matched_id == id)), ["smooth_x", "smooth_y", "dt"]] = [xy[0], xy[1], dt]
                motsynth.merged.loc[(( motsynth.merged.frame == row["frame"]) & (motsynth.merged.matched_id == id)), ["smooth_u", "smooth_v", "dt"]] = [uv[0], uv[1], dt]
                
            elif dt < 0:
                
                xy = KF_xy.step(row[["x", "y'"]].values, decrease_process_uncertainty=True)
                uv = KF_uv.step(row[["bb_center", "bb_bottom"]].values, decrease_process_uncertainty=True)
                motsynth.merged.loc[(( motsynth.merged.frame == row["frame"]) & (motsynth.merged.id_x == id)), ["smooth_x", "smooth_y", "dt"]] = [xy[0], xy[1], dt]
                motsynth.merged.loc[(( motsynth.merged.frame == row["frame"]) & (motsynth.merged.id_x == id)), ["smooth_u", "smooth_v", "dt"]] = [uv[0], uv[1], dt]
                
                for t in np.arange(1, abs(dt)+1):
                    xy = KF_xy.predict(increase_process_uncertainty=True)[0]
                    uv = KF_uv.predict(increase_process_uncertainty=True)[0]
                    if len(motsynth.merged.loc[(( motsynth.merged.frame == (row["frame"] + t)) & ( motsynth.merged.id_x == id))]) == 0: 
                        break
                    motsynth.merged.loc[(( motsynth.merged.frame == (row["frame"] + t)) & ( motsynth.merged.id_x == id)), ["pred_x", "pred_y", "gap"]] =  [xy[0], xy[1], abs(dt)]  
                    motsynth.merged.loc[(( motsynth.merged.frame == (row["frame"] + t)) & ( motsynth.merged.id_x == id)), ["pred_u", "pred_v"]] =  [uv[0], uv[1]]  
                
        directory = os.path.join( motsynth.root_dir, "postprocessing", "evaluation", "{:03d}".format(sequence))
        try:
            os.makedirs(directory)
        except: 
            pass
    motsynth.merged.to_csv(os.path.join(directory, "eval.txt"))
        