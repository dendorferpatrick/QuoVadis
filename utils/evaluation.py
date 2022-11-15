from PIL.Image import merge
import numpy as np
from numpy.lib.npyio import mafromtxt
from loader import CoordinateTransformer
import argparse
import pandas as pd
from kalman_filter import KalmanSmoother
import pickle
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore')

def coordinates_to_df(coordinates):
    df = pd.DataFrame(columns =["frame", "id", "x", "y"])
    for frame, data in tqdm(coordinates.items()) :
     
        for id, xyz in data.items():
           
            df.loc[len(df)] = [ frame, id , xyz[0], xyz[1]]
    df[["frame", "id"]] = df[["frame", "id"]].astype("int")
    return df


class Evaluator():
    def __init__(self, sequence): 
        self.sequence = sequence
        self.coordinate_transformer = CoordinateTransformer(sequence = sequence)

        bb_file = self.coordinate_transformer.file_loader.panoptic_bb_file()
        self.bb_df = pd.read_csv(bb_file)
        self.data = self.coordinate_transformer.sequence_loader.data
        # with open(r"/usr/wiss/dendorfp/dvl/projects/TrackingMOT/notebooks/{}.pkl".format(self.sequence), "rb") as input_file:
        #     data = pickle.load(input_file)
        # coordinates, inv_transformations, focal_lengths = data.values()
        # self.coordinate_df = coordinates_to_df(coordinates)
        # print(self.coordinate_df)

        # self.bb_df = pd.merge(self.bb_df , self.coordinate_df,  on=['frame', 'id'], how='left')

    def compute_detection_metrics(self):
        self.bb_df[self.bb_df.matched_id.isnull() == False] = self.bb_df[self.bb_df.matched_id.isnull() == False].astype('int32')
        merged_matches_left = pd.merge(self.data, self.bb_df, left_on=["frame", "id"] , right_on=["frame", "matched_id"], how="left")
        merged_matches_left["TP"] = (merged_matches_left.matched_id.isnull() == False) * 1.

        merged_matches_right = pd.merge(self.data, self.bb_df, left_on=["frame", "id"] , 
        right_on=["frame", "matched_id"], how="right")
        
        merged_matches_right["TP"] = (merged_matches_right.id_x.isnull() == False) * 1.
        

        print(merged_matches_left.groupby(pd.cut(merged_matches_left["visibility"], np.arange(0, 1.0+0.09, 0.10)))["TP"].mean())

        print("Precision", merged_matches_right.TP.mean())
    def analyse_visibility(self):
        
        
  
        matched_bb_df = self.bb_df[self.bb_df.matched_id.isnull() == False]
       
        matched_bb_df.sort_values(["matched_id", "frame"], inplace=True)
      
       
        matched_bb_df["partition"] = -1
        ids = matched_bb_df.matched_id.unique()
        partition = 0 
        
        
        matched_bb_df["dt"]  = -1
        for id in ids:
            id_matched = matched_bb_df[matched_bb_df.matched_id == id]
            id_matched.sort_values("frame")
      
            id_matched["dt"] = id_matched["frame"].diff(periods=-1).fillna(value=-1).astype("int").values
            partition+=1 
            for index, row in id_matched.iterrows(): 
                matched_bb_df.loc[(
                    (matched_bb_df.frame == row["frame"] ) & 
                    (matched_bb_df.matched_id == id)), ["partition", "dt"]] = [partition, row["dt"]]
                if row["dt"] != -1:
                    partition+=1 
                    
                # print(row["dt"] , partition)
            
             
        
        merged_matches_left = pd.merge(self.data,matched_bb_df, 
        left_on=["frame", "id"] , right_on=["frame", "matched_id"], how="left")
        
        

        matched_bb_df[["pred_x", "pred_y", 
        "smooth_x", "smooth_y",
        "pred_u", "pred_v", 
        "smooth_u", "smooth_v", 
        ]] = 0 
        # merged_matches_left
        matched_bb_df.to_csv("complete_{}.pkl".format(self.sequence))
    def predict_linear(self):
        matched_bb_df = pd.read_csv("complete_{}.pkl".format(self.sequence))
        matched_bb_df = matched_bb_df[matched_bb_df.partition.isnull() == False]
        current_id = None 
        matched_bb_df["u"] = matched_bb_df["bb_left"] + matched_bb_df["bb_width"]/ 2 
        matched_bb_df["v"] = matched_bb_df["bb_top"] + matched_bb_df["bb_height"]
        for partition in matched_bb_df.partition.unique():

            row  = matched_bb_df.loc[matched_bb_df.partition == partition, ["matched_id", "x", "y", "u", "v"]].values[0]
            
            new_id = row[0]
            if current_id != new_id: 
                current_id = new_id
                KF_xy = KalmanSmoother(np.array([row[1], row[2], 0.0 ,0.]))
                KF_uv = KalmanSmoother(np.array([row[3], row[4], 0.0 ,0.]))
                print("new kalman")
                gap = 0 
            
            
            trajectory_xy = matched_bb_df.loc[matched_bb_df.partition == partition, ["x", "y"]].values
            trajectory_uv = matched_bb_df.loc[matched_bb_df.partition == partition, ["u", "v"]].values
            M_sxsy, M_xy = KF_xy.smooth(trajectory_xy)
            M_susv, M_uv = KF_uv.smooth(trajectory_uv)
            time_dt = matched_bb_df.loc[matched_bb_df.partition == partition, ["frame", "dt", "partition"]].values
            matched_bb_df.loc[matched_bb_df.partition == partition, "smooth_x"] = M_sxsy[:, 0]
            matched_bb_df.loc[matched_bb_df.partition == partition, "smooth_y"] = M_sxsy[:, 1]
            matched_bb_df.loc[matched_bb_df.partition == partition, "pred_x"] = M_xy[:, 0]
            matched_bb_df.loc[matched_bb_df.partition == partition, "pred_y"] = M_xy[:, 1]

            matched_bb_df.loc[matched_bb_df.partition == partition, "smooth_u"] = M_susv[:, 0]
            matched_bb_df.loc[matched_bb_df.partition == partition, "smooth_v"] = M_susv[:, 1]
            matched_bb_df.loc[matched_bb_df.partition == partition, "pred_u"] = M_uv[:, 0]
            matched_bb_df.loc[matched_bb_df.partition == partition, "pred_v"] = M_uv[:, 1]
            matched_bb_df.loc[matched_bb_df.partition == partition, "gap"] = gap
            
            gap = 0
            for t in np.arange(1, abs(time_dt[-1, 1])): 
                gap+=1
                xy = KF_xy.predict(increase_process_uncertainty=True)
                uv = KF_uv.predict(increase_process_uncertainty=True)
                print(matched_bb_df.columns)
                matched_bb_df.loc[len(matched_bb_df)] = [None,  None, (time_dt[-1, 0] + t), 
                current_id, None, None, None, None, current_id, None, None, None, None,xy[0, 0], xy[0, 1], None, None, 
                 None, None,None, None, uv[0, 0], uv[0, 1], None]

        matched_bb_df.to_csv("prediction_{}.csv".format(self.sequence))





if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--sequence',type=str,
    #                     help='sequence')
    
    # args = parser.parse_args()
    # evaluator  = Evaluator(args.sequence)
    # # evaluator.analyse_visibility()
    # evaluator.predict_linear() 

    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    import numpy as np
    data = pd.read_csv("prediction_MOT16-02.csv")
    warnings.simplefilter(action='ignore')

    ids = data.matched_id.unique() 

    df_interest = data[data.dt != -1]
    # for (index, row) in df_interest.iterrows():
    #     print(row[["frame", "id", "x", "y", "pred_x", "pred_y"]])
    print(df_interest.columns)
    error_xy = []
    error_uv = []
    TP_xy = [] 
    TP_uv = [] 
    for id in ids:
        df_id = data[data.matched_id == id]
        partitions = df_id.partition.unique()

        for index, partition in enumerate(partitions):
            if index < 1:
                continue
            try:
                df_partition = df_id[df_id.partition == partition]
                row = df_partition[["frame", "x", "y","smooth_x", "smooth_y", "pred_x", "pred_y","dt", "partition", "gap"]].values
                if len(row) == 0:
                    continue
                
                    
                # if row[0, -1] < 100.:
                #     continue

        
                allPositions = data.loc[((data.frame == row[0, 0] )& (data.smooth_x.isnull() == False) ), ["smooth_x", "smooth_y", "matched_id"]].values
                d = allPositions[:, :-1] - row[:1, 5:7]

                distance = np.sqrt(np.sum(d**2, 1))
                min_id = np.argmin(distance)
                detection_id = allPositions[min_id, -1]
                TP_xy.append(detection_id == id)
                error_xy.append(np.array([[np.sqrt((row[0, 1] - row[0, 5])**2 + (row[0, 2] - row[0, 6])**2), row[0, -1], (detection_id == id) *1.]]))
                row = df_partition[["frame", "u", "v","smooth_u", "smooth_v", "pred_u", "pred_v","dt", "partition", "gap"]].values
                if len(row) == 0:
                    continue
                
                    
        #         if row[0, -1] < 20.:
        #             continue

                
                allPositions = data.loc[((data.frame == row[0, 0] )& (data.smooth_x.isnull() == False) ), ["smooth_u", "smooth_v", "matched_id"]].values
                d = allPositions[:, :-1] - row[:1, 5:7]

                distance = np.sqrt(np.sum(d**2, 1))
                min_id = np.argmin(distance)
                detection_id = allPositions[min_id, -1]
                TP_uv.append(detection_id == id)
                
                error_uv.append(np.array([[np.sqrt((row[0, 1] - row[0, 5])**2 + (row[0, 2] - row[0, 6])**2), row[0, -1], (detection_id == id) *1.]]))
            except: 
                pass
        

    print("ACCURACY XY {}".format(np.mean(TP_xy)))
    print("ACCURACY UV {}".format(np.mean(TP_uv)))
    error_array_xy = np.concatenate(error_xy)
    error_array_uv = np.concatenate(error_uv)
                            
    # plt.plot(error_array_xy[:, 1], error_array_xy[:, 0], ".")
    # plt.show()
                            
    # plt.plot(error_array_uv[:, 1], error_array_uv[:, 0], ".")
    # plt.show() 

    df_xy = pd.DataFrame(error_array_xy, columns=["error", "gap", "TP"])
    df_uv = pd.DataFrame(error_array_uv, columns=["error", "gap", "TP"])



    print(df_xy.groupby(pd.cut(df_xy["gap"], [0, 30, 60,  90, 120, 150, 240, 10000]))["TP"].mean())
    print(df_uv.groupby(pd.cut(df_uv["gap"], [0, 30, 60,  90, 120, 150, 240, 10000]))["TP"].mean())

    print(df_xy.groupby(pd.cut(df_xy["gap"],[0, 30, 60,  90, 120, 150, 240, 10000]))["TP"].count())
    print(df_uv.groupby(pd.cut(df_uv["gap"],[0, 30, 60,  90, 120, 150, 240, 10000]))["TP"].count())


    print(df_xy.groupby(pd.cut(df_xy["gap"],[0, 30, 60,  90, 120, 150, 240, 10000]))["TP"].count().sum())