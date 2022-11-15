import os
import glob 
import numpy as np
import matplotlib.pyplot as plt
import cv2



data_dir = "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/data/MOT20-*" 
sequences = glob.glob(data_dir)

print(sequences)

for seq_folder in sequences:
    seq = seq_folder.split("/")[-1]
    
    os.makedirs("/storage/user/dendorfp/MOT20/depth_img/{}".format(seq), exist_ok = True)
    ref_image =   "/storage/user/dendorfp/MOT20/img1/{}/img1/000001.jpg".format(seq) 
    img = plt.imread(ref_image)
    depth_files = glob.glob("/storage/user/dendorfp/MOT20/img1/{}/depth/".format(seq) + "/*.npz")
    print(img.shape)
    print(depth_files)
    for file in depth_files:
        depth = np.load(file)["arr_0"]
        new_image = cv2.resize(depth, dsize=(img.shape[1], img.shape[0]) , interpolation=cv2.INTER_LANCZOS4)
        img_file = file.split("/")[-1].split(".")[0] + ".npy"

        
        np.save("/storage/user/dendorfp/MOT20/depth_img/{}/".format(seq) + img_file,np.array(new_image))
        
        
    
    



