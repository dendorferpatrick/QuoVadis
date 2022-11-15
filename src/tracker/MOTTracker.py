from abc import ABC, abstractmethod
import pandas as pd
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(
            "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/Predictors/KalmanFilter")
from kalman_filter import KalmanFilter # noqa: E2

from .Tracker import Tracker
    
class MOTTracker(Tracker):
    def __init__(self, df = None, 
    df_detection = None, 
    tracking_file = None , 
    df_pos = None, 
    position_file = None, 
    clean_positions = True, 
    tracker_name = "CenterTrack", 
    detections = True, 
    tracker = True, 
    homography = "homography_depth", 
    visual_features = None):
        self.homography = homography
        self.df_pos = None
        if tracker:
            self.name = tracker_name
            if df is None:
                self.df = pd.read_csv(tracking_file)
            else: self.df = copy.deepcopy(df)
            assert tracker_name in self.df.tracker.unique(), f"Tracker `{tracker_name}` does not exist"
            self.df = self.df[self.df.tracker == tracker_name]
        
        else:
            self.df = df_detection
            self.name = "detections"
        if detections:
            self.df["id"] = np.arange(1, len(self.df) + 1) 


        # if df_pos is not None:
        #     self.df_pos = copy.deepcopy(df_pos)
        # elif position_file is not None:
        #     self.df_pos = pd.read_csv(position_file)
        # if self.df_pos is not None:
        #     self.df = self.df.merge(self.df_pos[["frame", "id", "x_world", "y_world", "z_world"]],
        #                                 left_on=["frame", "gt_id"], right_on=["frame", "id"], how="left")
        #     self.df["id"] = self.df["id_x"]


        if clean_positions: 
            print("clean position")
            self.df.sort_values("frame", inplace = True)
            for id in self.df.id.unique():

                tracker_df_id = self.df[self.df.id == id]
                trajectory_list = [] 
                frame = tracker_df_id.frame.min()
                mask = []
                occluded = []
                position = {}
                for index, row in tracker_df_id.iterrows():
                    
                    while frame < row.frame:
                        frame+=1
                        mask.append(False)
                        occluded.append(False)
                        trajectory_list.append([np.nan, np.nan])

                    if row.frame == frame: 
                        position[row.frame] = np.array([row.x, row.z])
                        
                        # for j, k in enumerate(range(row.frame - 30, row.frame +1)):
                        
                        #     if not k in position.keys():
                        #         continue
                        #     else:
                        #         v = (position[row.frame] - position[k]) * 30/(30-j+1e-10)
                        #         break
                       

                        trajectory_list.append([row.x, row.z])
                        mask.append(True)
                        if row.occluded == 1:
                            occluded.append(True)
                        else: occluded.append(False)
                        frame = row.frame
                    else: assert False
                trajectories = np.array(trajectory_list)
                if len(trajectories) < 2:
                    continue
                initial_state = np.concatenate(
                    (trajectories[0], np.array([0.,  0.])))
                kf = KalmanFilter( dt = 1/30,
                        initial_state=initial_state, 
                        measurement_uncertainty=0.1,
                        process_uncertainty=0.1, 
                        frame = 1)
                
                x, _ = kf.smooth(trajectories[1:], occluded = occluded[1:])
                xy= np.concatenate((initial_state[np.newaxis, :2], x),0) 
                # plt.plot(trajectories[:, 0], trajectories[:, 1], ".")
                # plt.plot(xy[:, 0], xy[:, 1], )
                # plt.show() 
                self.df.loc[self.df.id == id, ["x", "z"]] = xy[mask]
                
        if visual_features is not None:
            self.visual_features = np.load(visual_features)
        else: self.visual_features = None
        self.df["x_world"] = self.df.x
        self.df["y_world"] = self.df.z
        self.df["z_world"] = self.df.y


        self.df[["frame", "id"]] = self.df[["frame", "id"]].astype(int)

        self.get_pixel_coordinates() 

class CenterTrack(Tracker):
    def __init__(self, df = None , tracking_file = None , df_pos = None, 
                position_file = None, max_age = None, score = None, 
                tracker_name = None, model_name = None):
        self.name = tracker_name
        
        if df is None:
            self.df = pd.read_csv(tracking_file)
        else: self.df = copy.deepcopy(df)
        
        if tracker_name is not None:
            self.df = self.df[self.df.tracker == tracker_name]

        if model_name is not None:
            self.df = self.df[self.df.model == model_name]

        if df_pos is None:
            self.df_pos = pd.read_csv(position_file)
        else: self.df_pos = copy.deepcopy(df_pos)


        if max_age: 
            self.df = self.df[self.df.age <= max_age]
        
        if score:
            self.df = self.df[self.df.age > max_age]

        
        self.df = self.df.merge(self.df_pos[["frame", "id", "x_world", "y_world", "z_world"]],
                                  left_on=["frame", "gt_id"], right_on=["frame", "id"], how="left")
        self.df["id"] = self.df["id_x"]
     
        self.get_pixel_coordinates() 
        
class GTTracker(Tracker):

    def __init__(self, df = None,  gt_file = None, min_vis = 25): 
        assert min_vis in [25, 50, 75]
        self.name = f"gt_{min_vis}"
        if df is None:
            self.df = pd.read_csv(gt_file)
        else: self.df = df

        self.df["gt_id"] = self.df["id"]
        self.df["idsw"] = 0
        
        if min_vis is not None:
            self.df = self.df[self.df.visibility > (min_vis/ 100.)]
        self.df.sort_values(["id", "frame"], inplace = True)

        self.df["diff_t"] =  self.df.groupby("id")["frame"].diff(1)

        self.df["valid_step"] = ( self.df["diff_t"] != 1)*1
        idx =  self.df.groupby(["id"])['frame'].transform(
                            min) ==  self.df['frame']
        self.df[idx]["valid_step"] = 1
        self.df["id"] =  self.df.valid_step.cumsum()
        
        self.get_pixel_coordinates() 

    def create_tracker_output(self, output_filename):
        df = self.df
        df = df[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]]
        df["active"] = 1
        df.to_csv(output_filename, index = False)







