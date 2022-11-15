from kalman_filter import KalmanSmoother
from tqdm import tqdm
import warnings
from MOTSynth import MOTSynth
import numpy as np
import os
import pandas as pd
warnings.simplefilter(action='ignore')

import argparse

parser = argparse.ArgumentParser(description='Process sequences')
parser.add_argument('--sequence',type=int)

args = parser.parse_args()


sequence = args.sequence
df = pd.read_csv("/storage/user/dendorfp/MOTSynth/postprocessing/evaluation/{:03d}/eval.txt".format(sequence))
df_gaps = df[((pd.isnull(df.pred_x) == False ) & (pd.isnull(df.smooth_x) == False ))]

TP_xy = []
TP_uv = []
gap_list = []
for index, g_row in df_gaps.iterrows():

    allPositions = df.loc[((df.frame == g_row.frame )& (pd.isnull(df.matched_id) == False) ), ["x", "y'", "matched_id"]].values
    position = g_row[["pred_x", "pred_y"]].values

    d = allPositions[:, :-1] - position
    distance = np.sqrt(np.sum(d**2, 1))
    min_id = np.argmin(distance)
    detection_id = allPositions[min_id, -1]

    TP_xy.append((detection_id ==  g_row.id_x) * 1.)
    
    allPositions = df.loc[((df.frame == g_row.frame )& (pd.isnull(df.matched_id) == False) ), ["bb_center", "bb_bottom", "matched_id"]].values
    position = g_row[["pred_u",  "pred_v"]].values

    d = allPositions[:, :-1] - position
    distance = np.sqrt(np.sum(d**2, 1))
    min_id = np.argmin(distance)
    detection_id = allPositions[min_id, -1]

    TP_uv.append((detection_id ==  g_row.id_x) * 1.)
    gap_list.append(g_row.gap)

results = pd.DataFrame()
results["TP_xy"] = TP_xy
results["TP_uv"] = TP_uv
results["gap"] = gap_list


directory = os.path.join( "/storage/user/dendorfp/MOTSynth/results", "{:03d}".format(sequence))
try:
    os.makedirs(directory)
except: 
    pass

results.to_csv(os.path.join(directory, "results.txt"))
print("Stored results to {}".format(os.path.join(directory, "results.txt")))