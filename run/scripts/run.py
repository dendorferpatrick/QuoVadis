import argparse
import copy
import sys
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from datasets import MOTTracking
from tracker import MOTTracker
from TrajectoryPredictor import (KalmanFilterPredictor, LinearPredictor,
                                 MGGANPredictor, MotionModel,
                                 MultimodalLinearPredictor, OraclePredictor,
                                 StaticPredictor, get_y0, pix2real)

print("all good so far")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--tracker-name",
        type=str,
        default="CenterTrack",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--run_eval",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--save_results",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--plot_results",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--save_vis",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--make_video",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="MOT16-02",
        help="",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


args = get_parser().parse_args()
with open("config/base.yaml", "r") as stream:
    try:
        cfg_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def parse_sequence_arg(args):
    sequence_str = args.sequence
    challenge, sequence_nr_str = sequence_str.split("-")
    sequence_nr = int(sequence_nr_str)
    return challenge, sequence_nr


new_args = {}
assert len(args.opts) % 2 == 0, "Not valid number of opts inputs"
opts = args.opts
for i in np.arange(0, len(args.opts), 2):
    option = opts[i]
    assert option in cfg_dict, f'Additional option "{option}" not a valid key. Option does not exist'
    try:
        value = type(cfg_dict[option])(opts[i + 1])
    except ValueError as exc:
        raise ValueError(
            f'Value given in --opts for "{option}" is not valid. Make sure to insert value of type {type(cfg_dict[option])}.')
    cfg_dict[option] = value

cfg_dict.update(vars(args))

cfg_dict["challenge"], cfg_dict["sequence_nr"] = parse_sequence_arg(args)
cfg = argparse.Namespace(**cfg_dict)
print(cfg)

homography = "homography_depth"
egomotion = False
sequence_nr = 2

print(sequence_nr)
scale = 0.1
fields = ["rgb",
          "dets",
          #                                 "pose",
          "tracker",
          #                                 "lidar",
          #                                 "calibration",
          #                                 "depth",
          "map_img",
          homography,
          "panoptic",
          "egomotion",
          ]

mot = MOTTracking(partition=["{:02d}".format(cfg.sequence_nr)],
                  challenge=cfg.challenge,
                  fields=fields)
sequence = mot.data.sequences[0]
frame = 1
ax = sequence.plot_rgb(frame, show=False)
sequence.plot_dets(frame, ax=ax)


tracker_df = mot.data.sequences[0].tracker

item = sequence.__getitem__(1, fields=[homography, "map_img"])


visibility = item["map_img"]["visibility"]
height, width = visibility.shape
tracker_df["u"] = tracker_df["bb_left"] + 1/2.*tracker_df["bb_width"]
tracker_df["v"] = tracker_df["bb_top"] + tracker_df["bb_height"]
pixel_positions_raw = tracker_df[["u", "v", "frame"]].values
occluded = visibility[np.clip(pixel_positions_raw[:, 1].astype(
    int), 0, height - 1), np.clip(pixel_positions_raw[:, 0].astype(int),  0, width - 1)] == 0
tracker_df["occluded"] = occluded * 1


figy, ay = plt.subplots()
figx, ax = plt.subplots()
#     ay.plot(pixel_positions[:, 1], pos[:,1] , ".", label ="normal")


pixel_positions = np.concatenate(
    (pixel_positions_raw[:, :2], np.ones((len(pixel_positions_raw), 1))), 1)

if egomotion:
    frames = pixel_positions_raw[:, -1].astype(int)
    new_pos = np.zeros_like(pixel_positions)
    for frame in tqdm(np.unique(frames)):

        item = sequence.__getitem__(frame, [homography, "egomotion"])

        H = np.array(item[homography]["IPM"])

        ego_m = item["egomotion"]

        y0 = get_y0(H, width)

        pos = H.dot(pixel_positions[frames == frame].T).T
        pos = pos/pos[:, -1:]
        try:
            if y0 is not None:

                pos_t = pix2real(
                    H, pos*1., pixel_positions[frames == frame, :2]*1., y0, img_width=width)
            else:
                pos_t = pos*1.
            offset = np.array(H).dot(np.array([[int(width/2), height, 1]]).T).T
            offset = offset/offset[:, -1:]
            new_pos[frames == frame, :2] = pos_t[:, :2] - \
                ego_m["median"][np.newaxis] - offset[:, :2]
            if np.sum(np.isnan(new_pos[frames == frame, :2]) > 0):
                print(H)
                print(ego_m["median"])
        except:
            print(traceback.print_exc())
            print(H)
            print(pos)
            dsds

#             ax = sequence.plot_rgb(frame, show = False)


else:
    H = np.array(item[homography]["IPM"])
    y0 = get_y0(H, width)
    pos = H.dot(pixel_positions.T).T
    pos = pos/pos[:, -1:]

    if y0 is not None:
        new_pos = pix2real(H, pos*1., pixel_positions*1., y0, img_width=width)
    else:
        new_pos = pos*1.

tracker_df[["x", "z", "y"]] = new_pos

model_name = "3_long"


tracker = MOTTracker(df=tracker_df,
                     tracker_name=cfg.tracker_name,
                     tracker=True,
                     df_detection=None,
                     homography=homography,
                     detections=False,
                     clean_positions=False,
                     visual_features=f"{cfg.DATA_DIRECTORY}/{cfg.challenge}/tracker/{cfg.tracker_name}/features/{cfg.challenge}-{sequence_nr:02d}.npy")


frames = range(tracker.df.frame.min(), tracker.df.frame.max() + 1)
frames = range(tracker.df.frame.min(), tracker.df.frame.min() + 50)

motion_dim = 3

save_name = f"test_github"


def get_predictor(cfg):
    if cfg.PRED_MODEL == "linear":
        predictor = LinearPredictor()
    elif cfg.PRED_MODEL == "kalman":
        predictor = KalmanFilterPredictor(dt=1/30.,
                                          measurement_uncertainty_x=0.1,
                                          measurement_uncertainty_y=0.1,
                                          process_uncertainty=0.1)
    elif cfg.PRED_MODEL == "static":
        predictor = StaticPredictor()
    elif cfg.PRED_MODEL == "mggan":
        predictor = MGGANPredictor(
            model="3_long",
            nr_predictions=cfg.NR_PREDICTIONS,
            dataset_name="motsynth",
            pred_len=15,
            dt=cfg.DT
        )
    elif cfg.PRED_MODEL == "gan":
        predictor = MGGANPredictor(
            model="1_long",
            nr_predictions=cfg.NR_PREDICTIONS,
            dataset_name="motsynth",
            pred_len=15,
            dt=cfg.DT
        )
    else:
        raise ValueError(
            "No valid prediction model given for option 'PRED_MODEL'.")
    return predictor


predictor = get_predictor(cfg)


motion_model = MotionModel(predictor=predictor,
                           tracker=tracker,
                           sequence=sequence)
motion_model.run(motion_dim=motion_dim,
                 save_results=cfg.save_results,
                 save_name=save_name,
                 min_iou_threshold=cfg.MIN_IOU_THRESHOLD,
                 L2_threshold=cfg.L2_THRESHOLD,
                 IOU_threshold=cfg.IOU_THRESHOLD,
                 min_appearance_threshold=cfg.APP_THRESHOLD,
                 hallucinate=False,
                 frames=frames,
                 social_interactions=False,
                 Id_switch_metric="IOU",
                 reId_metric=cfg.REID_METRIC,
                 visibility=True,
                 max_age=cfg.MAX_AGE,
                 max_age_visible=cfg.MAX_AGE_VIS,
                 debug=True,
                 exists_ok=True,
                 y0=y0)
if cfg.save_results and cfg.run_eval:
    print("RUNNING EVALUATION")
    motion_model.run_eval()

if cfg.make_video or cfg.plot_results:
    motion_model.plot_results(show=cfg.plot_results, save=cfg.save_vis, make_video=cfg.make_video, save_folder="tmp",
                              frames=frames)
