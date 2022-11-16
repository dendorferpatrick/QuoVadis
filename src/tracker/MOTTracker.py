import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .Tracker import Tracker

from trajectory_predictor import KalmanFilter  


class MOTTracker(Tracker):
    def __init__(self, df=None,
                 df_detection=None,
                 tracking_file=None,
                 clean_positions=True,
                 tracker_name="CenterTrack",
                 detections=True,
                 tracker=True,
                 homography="homography",
                 visual_features=None):
        self.homography = homography
        if tracker:
            self.name = tracker_name
            if df is None:
                self.df = pd.read_csv(tracking_file)
            else:
                self.df = copy.deepcopy(df)
            assert tracker_name in self.df.tracker.unique(
            ), f"Tracker `{tracker_name}` does not exist"
            self.df = self.df[self.df.tracker == tracker_name]

        else:
            self.df = df_detection
            self.name = "detections"
        if detections:
            self.df["id"] = np.arange(1, len(self.df) + 1)

        if clean_positions:
            print("clean position")
            self.df.sort_values("frame", inplace=True)
            for id in self.df.id.unique():

                tracker_df_id = self.df[self.df.id == id]
                trajectory_list = []
                frame = tracker_df_id.frame.min()
                mask = []
                occluded = []
                position = {}
                for index, row in tracker_df_id.iterrows():

                    while frame < row.frame:
                        frame += 1
                        mask.append(False)
                        occluded.append(False)
                        trajectory_list.append([np.nan, np.nan])

                    if row.frame == frame:
                        position[row.frame] = np.array([row.x, row.z])
                        trajectory_list.append([row.x, row.z])
                        mask.append(True)
                        if row.occluded == 1:
                            occluded.append(True)
                        else:
                            occluded.append(False)
                        frame = row.frame
                    else:
                        assert False
                trajectories = np.array(trajectory_list)
                if len(trajectories) < 2:
                    continue
                initial_state = np.concatenate(
                    (trajectories[0], np.array([0.,  0.])))
                kf = KalmanFilter(dt=1/30,
                                  initial_state=initial_state,
                                  measurement_uncertainty=0.1,
                                  process_uncertainty=0.1,
                                  frame=1)

                x, _ = kf.smooth(trajectories[1:], occluded=occluded[1:])
                xy = np.concatenate((initial_state[np.newaxis, :2], x), 0)
          
                self.df.loc[self.df.id == id, ["x", "z"]] = xy[mask]

        if visual_features is not None:
            self.visual_features = np.load(visual_features)
        else:
            self.visual_features = None
        self.df["x_world"] = self.df.x
        self.df["y_world"] = self.df.z
        self.df["z_world"] = self.df.y

        self.df[["frame", "id"]] = self.df[["frame", "id"]].astype(int)

        self.get_pixel_coordinates()

