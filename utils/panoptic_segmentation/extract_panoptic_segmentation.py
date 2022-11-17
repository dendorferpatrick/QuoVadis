

from detectron2.utils.logger import setup_logger

setup_logger()

import argparse
import glob
import json
import os
import pickle
import random
import sys
from argparse import Namespace

import cv2
import matplotlib.pyplot as plt
# import some common libraries
import numpy as np
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
from tqdm import tqdm

sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/utils") 
sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT") 
import pandas as pd
from loader import PanopticLoader

cfg = get_cfg()

import itertools

import torch
from tqdm import tqdm

palette = list(itertools.product(np.arange(1,256), repeat=3)) 

step = 1000000

allColors = []
for i in range(step): 
    allColors.extend(palette[i::step])

meta_data_classes_pkl="/usr/wiss/dendorfp/dvl/projects/TrackingMOT/metaDataClasses.pkl"
with open(meta_data_classes_pkl, 'rb') as pkl_file:
    meta_data = Namespace(**pickle.load(pkl_file))

def run_panoptic_segmentation(frames, output_folder, sequence):

    # load pretrained panopitc segmentation model
    panoptic_segmentation_model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(panoptic_segmentation_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(panoptic_segmentation_model)

    
    np.random.shuffle(frames)
    panoptic_loader = PanopticLoader()
    
    predictor = DefaultPredictor(cfg)

    panoptic_output_folder = os.path.join(output_folder, "panoptic", sequence )
    segmentation_output_folder = os.path.join(output_folder, "segmentation", sequence )
    detection_output_folder = os.path.join(output_folder, "dets")
    for folder in [panoptic_output_folder, segmentation_output_folder, detection_output_folder]:
        if not os.path.isdir(folder):
            print("Creating Folder: {}" .format(folder))
            os.makedirs(folder)
    df_list = []
    save_det = os.path.join(detection_output_folder, "{}.txt".format(sequence))
    
    for f in tqdm(frames):
        im = cv2.imread(f)
        file = f.split("/")[-1]
        
        frame = int(file.split(".")[0])
        
        save_panoptic = os.path.join(panoptic_output_folder, "{:06d}.png".format(frame))
        save_seg = os.path.join(segmentation_output_folder, "{:06d}.png".format(frame))
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        panoptic_seg = panoptic_seg.cpu() 
        panoptic_img = torch.zeros(
                    panoptic_seg.size()[0], panoptic_seg.size()[1], 3).long()
       
       
        for info in (segments_info):
            id = info["id"]
            category_id = info["category_id"]
            is_thing = info["isthing"]
            if is_thing:
                color = meta_data.thing_colors[category_id]
            else:
                color = meta_data.stuff_colors[category_id]
            
            panoptic_img[panoptic_seg == id] = torch.Tensor(color).long()
        

        


        pixels, ids = panoptic_loader.get_pixels({"panoptic_seg": panoptic_seg, "segments_info": segments_info},  category = "pedestrian")
        pixels = pixels.astype("int")
        timesteps = np.ones(len(pixels))
        timesteps*= frame

        final_array = np.concatenate((timesteps[..., np.newaxis], ids[..., np.newaxis], pixels), 1)
        df_bb_t = pd.DataFrame(final_array, columns=["frame", "id", "u", "v"])
        dfuv = df_bb_t.groupby(["id"]).agg(min_u=('u', 'min'), 
                                        max_u = ('u' , 'max'), 
                                        min_v = ('v', 'min'), 
                                        max_v = ('v', 'max')).reset_index()
        dfuv["bb_width"] = dfuv["max_v"] - dfuv["min_v"]
        dfuv["bb_height"] = dfuv["max_u"] - dfuv["min_u"]
        dfuv["bb_left"] = dfuv["min_v"] 
        dfuv["bb_top"] = dfuv["min_u"]

        dfuv["frame"] = frame
        dfuv = dfuv.astype('int32')
     
        df_list.append(dfuv[["frame", "id" , 'bb_left', 'bb_top', 'bb_width', 'bb_height']])
        img_seg = np.zeros((im.shape[0], im.shape[1], 4))

        unique_ids = np.unique(ids)

        for id in unique_ids:
            pixel_id = pixels[ids == id]
            img_seg[pixel_id[:, 0], pixel_id[:, 1]] = (*allColors[id], 255)
        
        im_panoptic = Image.fromarray(np.array(panoptic_img).astype(np.uint8))
        im_panoptic.save(save_panoptic)      

        im_segmentation = Image.fromarray(np.array(img_seg).astype(np.uint8))
        im_segmentation.save(save_seg)          
        # with open(save_seg, 'wb') as handle:
        #     pickle.dump({"id": ids, "pixels": pixels}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df_full = pd.concat(df_list)
    df_full.sort_values(["frame", "id"], inplace = True)
    df_full.to_csv(save_det, index=False)
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--image_folder', default='',
                               help='folder containing images')
    parser.add_argument('--output_folder', default='',
                               help='output folder')
    parser.add_argument('--sequence' , default = '' , help="sequence")
    args = parser.parse_args()
    frames = glob.glob("{}/*.png".format(args.image_folder)) +  glob.glob("{}/*.jpg".format(args.image_folder))
    run_panoptic_segmentation(frames, args.output_folder, args.sequence)


