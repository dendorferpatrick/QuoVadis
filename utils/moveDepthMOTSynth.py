import os
import glob 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse


def moveMOTSynthDepth(seq):

   
    os.makedirs("/storage/user/dendorfp/MOTSynth/depth_img/{:03d}".format(seq), exist_ok = True)
 
    if seq > 100:
        return
    depth_files = glob.glob("/storage/user/dendorfp/MOT16/MOTSynth_depth/{:03d}/best_motsynth.pt".format(seq) + "/*.npz")
    
    for file in depth_files:
        depth = np.load(file)["arr_0"]
        # new_image = cv2.resize(depth, dsize=(1920, 1080) , interpolation=cv2.INTER_LANCZOS4)
        new_image = depth
        img_nr = int(file.split("/")[-1].split(".")[0]) + 1
        img_file = "{:06d}.npy".format(img_nr)

        
        np.save("/storage/user/dendorfp/MOTSynth/depth_img/{:03d}/".format(seq) + img_file,np.array(new_image))
        
        
    
    
if __name__ == "__main__": 
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required = False, type=int, 
                                help='challenge sequence')
    args = parser.parse_args() 
    moveMOTSynthDepth(args.sequence)
