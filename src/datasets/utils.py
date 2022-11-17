
import json
import os

import traceback
from gettext import translation

import cv2

import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
from tqdm import tqdm


def rot_x(theta):
    theta = np.atleast_1d(theta)
    n = len(theta)
    return np.stack(
        (
            np.stack([np.ones(n), np.zeros(n), np.zeros(n)], axis=-1),
            np.stack([np.zeros(n), np.cos(theta), -np.sin(theta)], axis=-1),
            np.stack([np.zeros(n), np.sin(theta), np.cos(theta)], axis=-1),
        ),
        axis=-2,
    )


def rot_y(theta):
    theta = np.atleast_1d(theta)
    n = len(theta)
    return np.stack(
        (
            np.stack([np.cos(theta), np.zeros(n), np.sin(theta)], axis=-1),
            np.stack([np.zeros(n), np.ones(n), np.zeros(n)], axis=-1),
            np.stack([-np.sin(theta), np.zeros(n), np.cos(theta)], axis=-1),
        ),
        axis=-2,
    )


def rot_z(theta):
    theta = np.atleast_1d(theta)
    n = len(theta)
    return np.stack(
        (
            np.stack([np.cos(theta), -np.sin(theta), np.zeros(n)], axis=-1),
            np.stack([np.sin(theta), np.cos(theta), np.zeros(n)], axis=-1),
            np.stack([np.zeros(n), np.zeros(n), np.ones(n)], axis=-1),
        ),
        axis=-2,
    )


def load_png(path: str):
    im = Image.open(path)
    # im = im.resize((960, 576), Image.ANTIALIAS)
    return np.array(im)


def oxts_to_pose_kitti(lat, lon, alt, roll, pitch, yaw):
    """This implementation is a python reimplementation of the convertOxtsToPose
    MATLAB function in the original development toolkit for raw data
    """
    n = len(lat)

    # converts lat/lon coordinates to mercator coordinates using mercator scale
    #        mercator scale             * earth radius
    scale = np.cos(lat[0] * np.pi / 180.0) * 6378137

    position = np.stack([
        scale * lon * np.pi / 180.,
        scale * np.log(np.tan((90. + lat) * np.pi / 360.)),
        alt,
    ], axis=-1)

    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    # extract relative transformation with respect to the first frame
    T0_inv = np.block(
        [[R[0].T, -R[0].T @ position[0].reshape(3, 1)], [0, 0, 0, 1]])
    T = T0_inv @ np.block([[R, position[:, :, None]],
                          [np.zeros((n, 1, 3)), np.ones((n, 1, 1))]])
    return T


def oxts_to_pose_motsynth(position, angles):
    """This implementation is a python reimplementation of the convertOxtsToPose
    MATLAB function in the original development toolkit for raw data
    """
    n = 1
    roll, pitch, yaw = np.radians(angles)

    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    T = np.block([[R, position[:, :, None]], [
                 np.zeros((n, 1, 3)), np.ones((n, 1, 1))]])
    T[:, 1, :] *= -1

    return T



def _get_frame_files(path):

    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _get_depth_files(path):
    path = os.path.join(path, "depth")
    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _get_depth_files_MOT(path):

    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _get_segmentation_files(path):

    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _get_panoptic_files(path):

    return sorted([int(os.path.splitext(file.name)[0]) for file in os.scandir(path)])


def _calibration_setup_cb_kitti(path):
    data = {}

    with open(path + ".txt") as f:

        data["P0"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape(
            (3, 4)
        )
        data["P1"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape(
            (3, 4)
        )
        data["P2"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape(
            (3, 4)
        )
        data["P3"] = np.fromstring(f.readline().split(":")[1], sep=" ").reshape(
            (3, 4)
        )

        line = f.readline()
        data["R_rect"] = np.fromstring(line[line.index(" "):], sep=" ").reshape(
            (3, 3)
        )

        line = f.readline()
        data["Tr_velo_cam"] = np.fromstring(
            line[line.index(" "):], sep=" "
        ).reshape((3, 4))

        line = f.readline()
        data["Tr_imu_velo"] = np.vstack([np.fromstring(
            line[line.index(" "):], sep=" "
        ).reshape((3, 4)),  [0, 0, 0, 1]])

        data["Tr_velo_imu"] = np.linalg.inv(data["Tr_imu_velo"])
    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = data["P1"][0, 3] / data["P1"][0, 0]
    T2 = np.eye(4)
    T2[0, 3] = data["P2"][0, 3] / data["P2"][0, 0]
    T3 = np.eye(4)
    T3[0, 3] = data["P3"][0, 3] / data["P3"][0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = data["Tr_velo_cam"]
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = data["P0"][0:3, 0:3]
    data['K_cam1'] = data["P1"][0:3, 0:3]
    data['K_cam2'] = data["P2"][0:3, 0:3]
    data['K_cam3'] = data["P3"][0:3, 0:3]

    return data, None


def _calibration_setup_cb_mot(path):
    # data = {}
    calibration = np.loadtxt(path + ".txt", delimiter=",")

    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        1,
        1,
        fx=calibration[0][0],
        fy=calibration[1][1],
        cx=calibration[0][2],
        cy=calibration[1][2],
    )

    return intrinsics, None


def _calibration_setup_cb_motsynth(path):

    width = 1920
    height = 1080
    fx = 1158
    fy = 1158

    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=fx,
        fy=fy,
        cx=width/2,
        cy=height/2,
    )

    return intrinsics, None


def _calibration_frame_cb(seq, key):
    return seq.calibration


def _homography_setup_cb(path): 
    
    with open(os.path.join(path,  "homography.json")) as json_file:
        homography = json.load(json_file)
    
    return homography, None
def _homography_frame_cb(seq, key): 
    homography = seq.homography
    if "-1" in homography:
        return homography['-1']
    else: 
        try: 
            return homography[str(key)]
        except: 
            keys = np.array([int(frame) for frame in homography.keys()])
            
            closest_key = np.argmin(keys - key)
            return homography[list(homography.keys())[closest_key]]



def _homography_bb_frame_cb(seq, key): 
    homography = seq.homography_bb
    if "-1" in homography:
        return homography['-1']
    else: 
        try: 
            
            return homography[str(key)]
        except: 
            
            keys = np.array([int(frame) for frame in homography.keys()])
            
            closest_key = np.argmin(keys - key)
            return homography[list(homography.keys())[closest_key]]
      


def _homography_gt_frame_cb(seq, key): 
    homography = seq.homography_gt
    if "-1" in homography:
        return homography['-1']
    else: 
        try: 
            return homography[str(key)]
        except: 
            keys = np.array([int(frame) for frame in homography.keys()])
            
            closest_key = np.argmin(keys - key)
            return homography[list(homography.keys())[closest_key]]
      

def _positions_h_frame_cb(seq, key):
    df = seq.positions_h
    return df[df.frame == key]

def _positions_h_depth_frame_cb(seq, key):
    df = seq.positions_h_depth
    return df[df.frame == key]

def _positions_h_depth_bb_frame_cb(seq, key):
    df = seq.positions_h_depth_bb
    return df[df.frame == key]

def _positions_h_gt_bb_frame_cb(seq, key):
    df = seq.positions_h_gt_bb
    return df[df.frame == key]

def _positions_h_gt_frame_cb(seq, key):
    df = seq.positions_h_gt
    return df[df.frame == key]
def _positions_h_setup_cb(path):

    df = pd.read_csv(path + ".txt")
    df[["0.10_x","0.10_y" ]] = df[["H_x", "H_y"]]

    return df,  None

def _positions_depth_setup_cb(path):

    df = pd.read_csv(path + ".txt")
    return df,  None

def _positions_depth_frame_cb(seq, key):
    df = seq.positions_depth
    return df[df.frame == key]

def _positions_setup_cb(path):

    df = pd.read_csv(path + ".txt")
    return df,  None


def _positions_frame_cb(seq, key):
    df = seq.positions
    return df[df.frame == key]


def _dets_setup_cb(path):
    
    df = pd.read_csv(path + "/det.txt")
    return df,  None


def _dets_frame_cb(seq, key):
    df = seq.dets
    return df[df.frame == key]

def _egomotion_cb(path):
    try:
        f = open(path + ".json")
        egomotion = json.load(f)
        f.close()
    
        frames = [int(frame) for frame in list(egomotion.keys())]
        frames = np.arange(1, np.max(frames) + 2)
        
        egomotion_dict = {}
    
        t_median = np.array([0., 0.])
        t_mean = np.array([0., 0.])
        R = np.eye(4)
        egomotion_dict[1] = {"R": R, "median" :t_median * 1., "mean" : t_mean * 1.}
        
        for frame in frames[1:]:
            t_median+= egomotion[str(frame -1)]["T_median"][:2]
            t_mean+= egomotion[str(frame -1)]["T_mean"][:2]

            egomotion_dict[frame] = {"median" :t_median * 1., "mean" : t_mean * 1., "R": np.linalg.inv(R)}
            
        return egomotion_dict, frames.tolist()
    except:
        print("Egomotion does not exist") 
        return [],  None
def _egomotion_frame_cb(seq, key): 
    if len(seq.egomotion) > 0:
        return seq.egomotion[key]
    else: 
        return None


def _tracker_setup_cb_mot(path):
    
    seq = path.split("/")[-2]

    path = "/".join(path.split("/")[:-2])
    
    trackers = os.listdir(path)
    
    df_list = []

    columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
               "category", "confidence", "visibility", "-"]
    for tracker in trackers:
        if tracker[0] == ".":
            continue

        model = "standard"

        tracker_path = os.path.join(
            path, tracker, "data", "{}.txt".format(seq))
        
        if not os.path.exists(tracker_path):
            continue
        df = pd.read_csv(tracker_path, names=columns)

        df["tracker"] = tracker
        df["model"] = model
        df["seq"] = seq

        try:
            idsw_path = os.path.join(
                "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/TrackEvalMOTSynth/output", model, "idsw", "{}.txt".format(seq))
            matches_path = os.path.join(
                "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/TrackEvalMOTSynth/output", model, "matches", "{}.txt".format(seq))

            idsw = pd.read_csv(idsw_path)
            idsw["idsw"] = 1

            df = df.merge(idsw, left_on=["frame", "id"], right_on=[
                "frame", "tracker_id"], how="left")
            df.drop(columns="tracker_id", inplace=True)
            matches = pd.read_csv(matches_path)
            matches["gt_id"] = matches["gt_id"].astype("int")

            df = df.merge(matches, left_on=["frame", "id"], right_on=[
                "frame", "tracker_id"], how="left")

            df.drop(columns="tracker_id", inplace=True)
            IOU_path = os.path.join(
                "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/TrackEvalMOTSynth/output", model, "IOU", "{}.txt".format(seq))

            df_IOU_raw = pd.read_csv(IOU_path)
            df_IOU = df_IOU_raw
            df_IOU["IOA"] = abs(df_IOU["IOA"])

            df_IOU = df_IOU.groupby(["frame", "id1"]).agg(
                {"IOU": "max", "IOA": "max"}).reset_index()

            df = df.merge(df_IOU[["frame", "id1", "IOU", "IOA"]], left_on=[
                "frame", "id"], right_on=["frame", "id1"], how="left")

            g = df_IOU_raw.groupby(["frame", "id1"]).cumcount()
            L = (df_IOU_raw.set_index(["frame", "id1", g])
                # .unstack(fill_value=0)
                .stack().groupby(["frame", "id1"])
                .apply(lambda x: np.reshape(x.values, (-1, 3)))
                ).reset_index()

            L["interaction"] = L[0]
            df.drop(columns="id1", inplace=True)
            df = df.merge(L[["frame", "id1", "interaction"]], left_on=[
                "frame", "id"], right_on=["frame", "id1"], how="left", suffixes=('', '_y'))
            df.drop(columns="id1", inplace=True)
        except:
            pass
        df_list.append(df)
    # except:
    #     print(traceback.format_exc())
    #     pass

    df = pd.concat(df_list)

    return df,  None


def _tracker_setup_cb(path):
    seq = path.split("/")[-1]
    path = "/" + os.path.join(*path.split("/")[:-1])

    trackers = os.listdir(path)

    df_list = []
    for tracker in trackers:
        if tracker == "archive":
            continue

        models = os.listdir(os.path.join(path, tracker))
        for model in models:
            if model == "archive":
                continue

            try:
                tracker_path = os.path.join(
                    path, tracker, model, "data",  "{}.txt".format(seq))
                df = pd.read_csv(tracker_path)

                if (model != "motSynth_fulltest_private_motsynth_no_age"):
                    continue

                df["tracker"] = tracker
                df["model"] = model
                df["seq"] = seq
                idsw_path = os.path.join(
                    path, tracker, model, "output", "normal",  "idsw", "{}.txt".format(seq))
                matches_path = os.path.join(
                    path, tracker, model, "output", "normal",  "matches", "{}.txt".format(seq))

                idsw = pd.read_csv(idsw_path)
                idsw["idsw"] = 1

                df = df.merge(idsw, left_on=["frame", "id"], right_on=[
                              "frame", "tracker_id"], how="left")
                df.drop(columns="tracker_id", inplace=True)
                matches = pd.read_csv(matches_path)
                matches["gt_id"] = matches["gt_id"].astype("int")

                df = df.merge(matches, left_on=["frame", "id"], right_on=[
                              "frame", "tracker_id"], how="left")

                df.drop(columns="tracker_id", inplace=True)
                # IOU_path = os.path.join(
                #     "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/TrackEvalMOTSynth/output", model, "IOU", "{}.txt".format(seq))

                # df_IOU_raw = pd.read_csv(IOU_path)
                # df_IOU = df_IOU_raw
                # df_IOU["IOA"] = abs(df_IOU["IOA"])

                # df_IOU = df_IOU.groupby(["frame", "id1"]).agg(
                #     {"IOU": "max", "IOA": "max"}).reset_index()

                # df = df.merge(df_IOU[["frame", "id1", "IOU", "IOA"]], left_on=[
                #               "frame", "id"], right_on=["frame", "id1"], how="left")

                # g = df_IOU_raw.groupby(["frame", "id1"]).cumcount()
                # L = (df_IOU_raw.set_index(["frame", "id1", g])
                #      # .unstack(fill_value=0)
                #      .stack().groupby(["frame", "id1"])
                #      .apply(lambda x: np.reshape(x.values, (-1, 3)))
                #      ).reset_index()

                # L["interaction"] = L[0]
                # df.drop(columns="id1", inplace=True)
                # df = df.merge(L[["frame", "id1", "interaction"]], left_on=[
                #               "frame", "id"], right_on=["frame", "id1"], how="left", suffixes=('', '_y'))
                # df.drop(columns="id1", inplace=True)

                df_list.append(df)
            except:
                print(traceback.format_exc())
                pass

    df = pd.concat(df_list)

    return df,  None


def _tracker_frame_cb(seq, key):
    df = seq.tracker

    return df[df.frame == key]


def _masks_frame_cb(seq, key):
    df = seq.masks
    return df[df.frame == key]


def _masks_setup_cb(path):
    cols = ["frame", "id", "a", "height", "width", "mask"]
    df = pd.read_csv(path + "/gt/gt.txt", sep=" ",  names=cols)
    return df, df.frame.unique().tolist()


def _labels_frame_cb(seq, key):
    df = seq.labels
    return df[df.frame == key]


def _labels_setup_cb_mot(path):
    cols = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
            "confidence", "class", "visibility", "x", "y", "z"]

    # return only class pedestrian

    df = pd.read_csv(path + "/gt/gt.txt", names=cols)
    df = df[df["class"].isin([1, 7])]
    df = df[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
             "confidence", "class", "visibility"]]
    return df, list(np.sort(df.frame.unique()))


def _labels_setup_cb(path):

    # cols = ("frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
    #         "category", "confidence", "visibility", "x", "z", "y"
    #         )
    df = pd.read_csv(path + "/gt/gt.txt")

    df_IOU_raw = pd.read_csv(path + "/IOU/IOU.txt")
    df_IOU = df_IOU_raw
    df_IOU["IOA"] = abs(df_IOU["IOA"])

    df_IOA_id1 = df_IOU.groupby(["frame", "id1"]).agg(
        {"IOA": "max"}).reset_index()
    df_IOA_id2 = df_IOU.groupby(["frame", "id2"]).agg(
        {"IOA": "max"}).reset_index()

    df_IOA = df_IOA_id1.merge(df_IOA_id2, left_on=["frame", "id1"], right_on=[
                              "frame", "id2"], how="inner")

    df_IOA["IOA_IO"] = df_IOA[["IOA_x", "IOA_y"]].values.max(axis=1)

    df_IOU = df_IOU.merge(df[["frame", "id", "visibility"]], left_on=[
                          "frame", "id1"], right_on=["frame", "id"], how="left")

    df_IOU.rename(columns={"visibility": "visibility_1"}, inplace=True)

    df_IOU = df_IOU.merge(df[["frame", "id", "visibility"]], left_on=[
                          "frame", "id2"], right_on=["frame", "id"], how="left")

    df_IOU.rename(columns={"visibility": "visibility_2"}, inplace=True)

    df_IOU["visibility_IO"] = df_IOU[[
        "visibility_1", "visibility_2"]].values.max(axis=1)

    df_max = df_IOU.groupby(["id1", "frame"]).agg(
        {'IOA': 'max', 'IOU': 'max', "visibility_IO": "max"}).reset_index()

    df = df.merge(df_max[["frame", "id1", "IOU", "IOA", "visibility_IO"]], left_on=[
                  "frame", "id"], right_on=["frame", "id1"], how="left")
    df.drop(columns="id1", inplace=True)
    df = df.merge(df_IOA[["frame", "id1", "IOA_IO"]], left_on=[
                  "frame", "id"], right_on=["frame", "id1"], how="left")
    df.drop(columns="id1", inplace=True)

    df[["IOU", "visibility_IO", "IOA_IO", "IOA"]] = df[[
        "IOU", "visibility_IO", "IOA_IO", "IOA"]].fillna(0)

    df_IOU = df_IOU_raw
    df_IOU["IOA"] = abs(df_IOU["IOA"])

    df_IOU = df_IOU.groupby(["frame", "id1"]).agg({"IOU": "max", "IOA": "max"}).rename(
        columns={"IOU": "IOU_max", "IOA": "IOA_max"}).reset_index()

    df = df.merge(df_IOU[["frame", "id1", "IOU_max", "IOA_max"]], left_on=[
                  "frame", "id"], right_on=["frame", "id1"], how="left")

    g = df_IOU_raw.groupby(["frame", "id1"]).cumcount()
    L = (df_IOU_raw.set_index(["frame", "id1", g])
         # .unstack(fill_value=0)
         .stack().groupby(["frame", "id1"])
         .apply(lambda x: np.reshape(x.values, (-1, 3)))
         ).reset_index()

    L["interaction"] = L[0]
    df.drop(columns="id1", inplace=True)
    df = df.merge(L[["frame", "id1", "interaction"]], left_on=[
                  "frame", "id"], right_on=["frame", "id1"], how="left", suffixes=('', '_y'))
    df.drop(columns="id1", inplace=True)

    df["status"] = 0
    alpha_vis = 0.25
    alpha_IOU = 0.25
    alpha_IOA = 0.25

    df.loc[((df.visibility >= alpha_vis) & (df.IOU < alpha_IOU)), "status"] = 1
    df.loc[((df.visibility >= alpha_vis) & (
        (df.IOA >= alpha_IOA) | (df.IOA_IO >= alpha_IOA))), "status"] = 2
    df.loc[((df.visibility < alpha_vis) & ((df.IOU < alpha_IOU)
            | (df.visibility_IO < alpha_vis))), "status"] = 4
    df.loc[((df.visibility < alpha_vis) & (
        (df.IOU >= alpha_IOU) | (df.IOA >= alpha_IOA))), "status"] = 3

    df["y_pixel"] = df["bb_top"] + df["bb_height"]
    df["x_pixel"] = df["bb_left"] + df["bb_width"] / 2
    return df, df.frame.unique().tolist()


def _pose_frame_cb(seq, key):
    return seq.pose[key]


def _pose_setup_cb_mot(path):

    rotations = np.load(path + "_floor_rotations.npy")
    translations = np.load(path + "_floor_translations.npy")

    # extract a relative pose with respect to the original frame
    poses_dict = {}
    for frame, (rot_mat, trans_vec) in enumerate(zip(rotations, translations)):
        rot_mat[:2] *= -1
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, -1] = -trans_vec

        poses_dict[frame + 1] = T

    return poses_dict, list(poses_dict.keys())


def _pose_setup_cb_motsynth(path):

    f = open(path + ".json")
    annotation = json.load(f)
    f.close()

    df_annotation = pd.DataFrame(annotation["images"])

    df_poses = df_annotation[["frame_n", "cam_world_pos", "cam_world_rot"]]
    df_poses["cam_world_pos"] = np.array(df_poses["cam_world_pos"])
    df_poses.sort_values("frame_n", inplace=True)

    # extract a relative pose with respect to the original frame
    poses_dict = {}
    for index, row in df_poses.iterrows():

        poses_dict[row.frame_n] = oxts_to_pose_motsynth(
            np.array([row.cam_world_pos]), row.cam_world_rot)[0]

    return poses_dict, df_annotation.frame_n.unique().tolist()


def _pose_setup_cb_kitti(path):
    # see readme of raw data dev kit for explanation of these fields
    cols = (
        "lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf",
        "vl", "vu", "ax", "ay", "az", "af", "al", "au", "wx", "wy",
        "wz", "wf", "wl", "wu", "posacc", "velacc", "navstat", "numsats",
        "posmode", "velmode", "orimode",
    )
    df = pd.read_csv(path + ".txt", sep=" ", names=cols, index_col=False)

    # extract a relative pose with respect to the original frame
    poses = oxts_to_pose_kitti(
        *df[["lat", "lon", "alt", "roll", "pitch", "yaw"]].values.T)
    return poses, df.index.tolist()


def _rgb_frame_cb(seq, key):

    frames = os.listdir(os.path.join(
        seq.prefix, seq.fields["rgb"].folder, seq.name))
    frame = frames[0]
    [number, ext] = frame.split(".")
    number_len = len(number)
    key = (number_len - len(str(key))) * "0" + str(key)
    path = os.path.join(
        seq.prefix, seq.fields["rgb"].folder, seq.name,  f"{key}.{ext}")
    return load_png(path)


def _rgb_setup_cb(path):
    return None, _get_frame_files(path)


def _rgb_frame_cb_MOT(seq, key):

    frames = os.listdir(os.path.join(
        seq.prefix, seq.fields["rgb"].location, seq.name, seq.fields["rgb"].folder))
    
    
    frame = frames[0]
    [number, ext] = frame.split(".")
    number_len = len(number)
    key = (number_len - len(str(key))) * "0" + str(key)
    path = os.path.join(
        seq.prefix,  seq.fields["rgb"].location,seq.name, "img1", f"{key}.{ext}")

    return load_png(path)


def _rgb_setup_cb_MOT(path):


    return None, _get_frame_files(path)


def _map_setup_cb(path):

    rgb_path = os.path.join(path,  f"rgb.png")
    classes_path = os.path.join(path,  f"classes.png")
    metadata_json = os.path.join(path,  f"mapping.json")
    visibility_path = os.path.join(path,  f"visibility.png")

    
    rgb = load_png(rgb_path)
    classes = load_png(classes_path)

    visibility_img = load_png(visibility_path)

    with open(metadata_json, 'r') as fp:
        metadata = json.load(fp)
    try:

        visibility_json = os.path.join(path,  f"visibility.json")
        with open(visibility_json, 'r') as fp:
            visibility = json.load(fp)
        return {"rgb": rgb, "classes": classes, "metadata": metadata, "visibility": visibility, "visibility_img": visibility_img}, [int(frame) for frame in visibility.keys()]
    except:
        return {"rgb": rgb, "classes": classes, "metadata": metadata, "visibility": {-1: None}, "visibility_img": visibility_img}, None

    


def _map_img_setup_cb(path):
    
    
    return None, None


def _map_frame_cb(seq, key):
    rgb, classes, metadata, visibility, visibility_img = seq.map.values()

    return {"rgb": rgb, "classes": classes, "metadata": metadata, "visibility": visibility[str(key)], "visibility_img": visibility_img}


def _map_img_frame_cb(seq, key):
    output = {}
    if os.path.exists(os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "rgb_{}.png".format(key))):
        rgb_path = os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "rgb_{}.png".format(key))
    else: 
        rgb_path = os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "rgb.png".format(key))
    if os.path.exists(os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "visibility_{}.png".format(key))):
        visibility_path = os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "visibility_{}.png".format(key))
    else: 
        visibility_path = os.path.join(
        seq.prefix,  seq.fields["map_img"].folder, seq.name, "visibility.png".format(key))

   
    visibility = load_png(visibility_path)
    rgb = load_png(rgb_path)
   

    return { "rgb": rgb , "visibility": visibility}



def _depth_frame_cb(seq, key):
    path = os.path.join(
        seq.prefix, seq.fields["depth"].folder, seq.name, "depth",  f"{key:04d}.png")
    return load_png(path)


def _depth_setup_cb(path):
    return None, _get_depth_files(path)


def _depth_frame_cb_MOT(seq, key):
    path = os.path.join(
        seq.prefix, seq.fields["depth"].folder, seq.name,  f"{key:06d}.npy")
    
    depth = np.load(path)
    
    return depth


def _depth_setup_cb_MOT(path):
    return None, _get_depth_files_MOT(path)


def _panoptic_frame_cb(seq, key):

    path = os.path.join(
        seq.prefix, seq.fields["panoptic"].folder, seq.name, f"{key:06d}.png")
    panoptic_png = load_png(path)
    # print(panoptic_png)

    class_img, mask = panoptic_loader.get_panoptic_img_MOTSynth(panoptic_png)

    return {"coco": panoptic_png,
            "category": class_img,
            "mask": mask,
            "panoptic_id": None}


def _panoptic_setup_cb(path):
    return None, _get_panoptic_files(path)


def _segmentation_frame_cb(seq, key):

    path = os.path.join(
        seq.prefix, seq.fields["segmentation"].folder, seq.name, f"{key:06d}.png")

    segmentation_png = load_png(path)

    return segmentation_png


def _segmentation_setup_cb(path):
    return None, _get_segmentation_files(path)


def load_lidar_data(path):
    scan = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    scan[:, -1] = 1.
    return scan


def _lidar_frame_cb_kitti(seq, key):
    try:
        scan = load_lidar_data(os.path.join(
            seq.prefix, seq.fields["lidar"].folder, seq.name, f"{key:06d}.bin"))
    except FileNotFoundError:
        # training sequence 1 has some frames without scans
        scan = np.empty((0, 4), dtype=np.float32)
    return scan


def _lidar_setup_cb_kitti(path):
    return None, _get_frame_files(path)


def _lidar_frame_cb_motsynth(seq, key):
    abs_min = 1008334389
    abs_max = 1067424357
    n = 1.04187
    f = 800
    path = os.path.join(
        seq.prefix, seq.fields["lidar"].folder, seq.name, "depth",  f"{key:04d}.png")
    depth = cv2.imread(path)[:, :, 0]

    depth = np.uint32(depth)

    depth = depth / 255
    depth = (depth * (abs_max - abs_min)) + abs_min
    depth = depth.astype('uint32')
    depth.dtype = 'float32'

    y = (-(n*f)/(n-f))/(depth-(n/(n-f)))
    y = y.reshape((1080, 1920))

    return y


def _lidar_setup_cb_motsynth(path):

    return None, _get_depth_files(path)


from copy import deepcopy
from math import sqrt

import numpy as np
import pandas as pd
from cv2 import threshold
from scipy.optimize import linear_sum_assignment


def compute_focal_length(x1: np.array, x2: np.array, u1, u2, z1, z2, max_distance=20):
    """ This function estimates the focal length of an image assuming a pinhole camera """
    distance = np.sqrt(np.sum((x1-x2)**2))
    if distance > max_distance:
        return False
    A = (u1[0]*z1 - u2[0]*z2)**2 + (u1[1]*z1 - u2[1]*z2)**2
    f = np.sqrt((A/distance**2))
    return f


# Implements Kabsch algorithm - best fit.
# Supports scaling (umeyama)
# Compares well to SA results for the same data.
# Input:
#     Nominal  A Nx3 matrix of points
#     Measured B Nx3 matrix of points
# Returns s,R,t
# s = scale B to A
# R = 3x3 rotation matrix (B to A)
# t = 3x1 translation vector (B to A)

def rigid_transform_3D(A, B, scale):
    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:

        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T
    R = R*c

    transformation = np.zeros((4, 4))
    transformation[3, 3] = 1
    transformation[:3, :3] = R
    transformation[:3, 3] = t.T
    return transformation


def calculate_similarities(bboxes1, bboxes2, img_shape = None, do_ioa = None):
    similarity_scores = _calculate_box_ious(
        bboxes1, bboxes2, box_format="xywh", img_shape = img_shape, do_ioa = do_ioa)
    return similarity_scores


def ious(bboxes1, bboxes2,  img_shape = None):
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
    Allows variable box formats ('xywh' and 'x0y0x1y1').
    If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
    used to determine if detections are within crowd ignore region.
    """

    # layout: (x0, y0, w, h)
    bboxes1 = deepcopy(bboxes1)
    bboxes2 = deepcopy(bboxes2)

    bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
    bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
    bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
    bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
    if img_shape is not None:
        bboxes1[:, (0, 2)] = np.clip(bboxes1[:, (0, 2)], 0, img_shape[1])
        bboxes1[:, (1, 3)] = np.clip(bboxes1[:, (1, 3)], 0, img_shape[0])
        bboxes2[:, (0, 2)] = np.clip(bboxes2[:, (0, 2)], 0, img_shape[1])
        bboxes2[:, (1, 3)] = np.clip(bboxes2[:, (1, 3)], 0, img_shape[0])

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1, bboxes2)
    max_ = np.maximum(bboxes1, bboxes2)

    
    intersection = np.maximum(
        min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)*1.
    
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * \
        (bboxes1[..., 3] - bboxes1[..., 1])

    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * \
        (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection
    intersection[area1 <= 0 + np.finfo('float').eps,] = 0
    intersection[ area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1
    ious = intersection / union
    return ious

def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False, img_shape = None):
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
    Allows variable box formats ('xywh' and 'x0y0x1y1').
    If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
    used to determine if detections are within crowd ignore region.
    """

    
    if box_format in 'xywh':
        # layout: (x0, y0, w, h)
        bboxes1 = deepcopy(bboxes1)
        bboxes2 = deepcopy(bboxes2)

        bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
        bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
        bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
        bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        if img_shape is not None:
            bboxes1[:, (0, 2)] = np.clip(bboxes1[:, (0, 2)], 0, img_shape[1])
            bboxes1[:, (1, 3)] = np.clip(bboxes1[:, (1, 3)], 0, img_shape[0])
            bboxes2[:, (0, 2)] = np.clip(bboxes2[:, (0, 2)], 0, img_shape[1])
            bboxes2[:, (1, 3)] = np.clip(bboxes2[:, (1, 3)], 0, img_shape[0])

    elif box_format not in 'x0y0x1y1':
        raise (Exception('box_format %s is not implemented' % box_format))

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(
        min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)*1.
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * \
        (bboxes1[..., 3] - bboxes1[..., 1])

    if do_ioa:
        ioas = np.zeros_like(intersection)
       
        valid_mask = area1 > 0 + np.finfo('float').eps
     
        ioas[valid_mask, :] = intersection[valid_mask, :] / \
            area1[valid_mask][:, np.newaxis]
        
        ioas*= (-1 +2 * (bboxes1[:,  np.newaxis , 3] <= bboxes2[ np.newaxis, :, 3]))
        return ioas
    else:
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * \
            (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
        intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
        intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
        intersection[union <= 0 + np.finfo('float').eps] = 0
        union[union <= 0 + np.finfo('float').eps] = 1
        ious = intersection / union
        return ious


def id_matching(data_1, data_2, img_shape = None):
    """ input data [frame, id, x, y,  w, t] """
    assert data_1.shape[-1] == 6, "data_1 wrong shape"
    assert data_2.shape[-1] == 6, "data_2 wrong shape"
    THRESHOLD = 0.5
    data_1 = data_1.astype(int)
    data_2 = data_2.astype(int)

    matches_list = [] 
    continguous_dict_1 = {}
    continguous_dict_2 = {}
    for new_id , id in enumerate(np.unique(data_1[:, 1])):
        data_1[data_1[:, 1] == id , 1] = new_id
        continguous_dict_1[new_id] = id
    for new_id , id in enumerate(np.unique(data_2[:, 1])):
        data_2[data_2[:, 1] == id , 1] = new_id
        continguous_dict_2[new_id] = id
    frames = np.unique(data_1[:, 0])
    
    # make id contingous

    num_1_ids = len(np.unique(data_1[:, 1]))
    # id_1_count = np.zeros(num_1_ids)  # For MT/ML/PT
    # matched_1_count = np.zeros(num_1_ids)  # For MT/ML/PT
    # frag_1_count = np.zeros(num_1_ids)  # For Frag

    # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
    # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
    prev_id = np.nan * np.zeros(num_1_ids)  # For scoring IDSW
    prev_timestep_id = np.nan * np.zeros(num_1_ids)  # For matching IDSW
    matches = {}

    for frame in frames:
        
        bboxes1 = data_1[data_1[:, 0] == frame]
        bboxes2 = data_2[data_2[:, 0] == frame]
        if len(bboxes2) == 0:
            continue
        if len(bboxes1) == 0:
            continue
        
        score_mat = calculate_similarities(
            bboxes1=bboxes1[:, 2:], bboxes2=bboxes2[:, 2:], img_shape = img_shape)

        ids_t_1 = bboxes1[:, 1]
        ids_t_2 = bboxes2[:, 1]
        
        # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily

        # score_mat = (ids_t_2[np.newaxis, :] ==
        #              prev_timestep_id[ids_t_1[:, np.newaxis]])

        # score_mat = 1000 * score_mat + similarity
        score_mat[score_mat < THRESHOLD - np.finfo('float').eps] = 0

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows,
                                          match_cols] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        matched_gt_ids = ids_t_1[match_rows]
        matched_tracker_ids = ids_t_2[match_cols]

        # Calc IDSW for MOTA
        prev_matched_tracker_ids = prev_id[matched_gt_ids]
        # is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
        #     np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))

        # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep

        prev_timestep_id[:] = np.nan
        prev_timestep_id[matched_gt_ids] = matched_tracker_ids
        matches[frame] = {continguous_dict_2[tracker_id]: continguous_dict_1[gt_id] for (
            tracker_id, gt_id) in zip(matched_tracker_ids, matched_gt_ids)}
        matches_list.extend([[frame, continguous_dict_2[tracker_id], continguous_dict_1[gt_id]] for (
            tracker_id, gt_id) in zip(matched_tracker_ids, matched_gt_ids)])
        
    return matches, matches_list


def id_matching_ioa(data_1, data_2, img_shape = None):
    """ input data [frame, id, x, y,  w, t] """
    assert data_1.shape[-1] == 6, "data_1 wrong shape"
    assert data_2.shape[-1] == 6, "data_2 wrong shape"
    THRESHOLD = 0.5
    data_1 = data_1.astype(int)
    data_2 = data_2.astype(int)

    matches_list = [] 
    continguous_dict_1 = {}
    continguous_dict_2 = {}
    for new_id , id in enumerate(np.unique(data_1[:, 1])):
        data_1[data_1[:, 1] == id , 1] = new_id
        continguous_dict_1[new_id] = id
    for new_id , id in enumerate(np.unique(data_2[:, 1])):
        data_2[data_2[:, 1] == id , 1] = new_id
        continguous_dict_2[new_id] = id
    frames = np.unique(data_1[:, 0])
    
    # make id contingous

    num_1_ids = len(np.unique(data_1[:, 1]))
    # id_1_count = np.zeros(num_1_ids)  # For MT/ML/PT
    # matched_1_count = np.zeros(num_1_ids)  # For MT/ML/PT
    # frag_1_count = np.zeros(num_1_ids)  # For Frag

    # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
    # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
    prev_id = np.nan * np.zeros(num_1_ids)  # For scoring IDSW
    prev_timestep_id = np.nan * np.zeros(num_1_ids)  # For matching IDSW
    matches = {}

    for frame in frames:
        
        bboxes1 = data_1[data_1[:, 0] == frame]
        bboxes2 = data_2[data_2[:, 0] == frame]
        if len(bboxes2) == 0:
            continue
        if len(bboxes1) == 0:
            continue
        
        score_mat = abs(calculate_similarities(
            bboxes1=bboxes1[:, 2:], bboxes2=bboxes2[:, 2:], img_shape = img_shape, do_ioa=True))
        
        ids_t_1 = bboxes1[:, 1]
        ids_t_2 = bboxes2[:, 1]
        
        # # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily

        # score_mat = (ids_t_2[np.newaxis, :] ==
        #              prev_timestep_id[ids_t_1[:, np.newaxis]])

        # score_mat = 1000 * score_mat + similarity
        score_mat[score_mat < THRESHOLD - np.finfo('float').eps] = 0

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows,
                                          match_cols] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        matched_gt_ids = ids_t_1[match_rows]
        matched_tracker_ids = ids_t_2[match_cols]

        # Calc IDSW for MOTA
       
        # is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
        #     np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))

        # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep

        prev_timestep_id[:] = np.nan
        prev_timestep_id[matched_gt_ids] = matched_tracker_ids
        matches[frame] = {continguous_dict_2[tracker_id]: continguous_dict_1[gt_id] for (
            tracker_id, gt_id) in zip(matched_tracker_ids, matched_gt_ids)}
        matches_list.extend([[frame, continguous_dict_2[tracker_id], continguous_dict_1[gt_id]] for (
            tracker_id, gt_id) in zip(matched_tracker_ids, matched_gt_ids)])
        
    return matches, matches_list


def overlap(data,  img_shape = None):
    """ input data [frame, id, x, y,  w, t] """
    assert data.shape[-1] == 6, "data_1 wrong shape"
   
    
    data = data.astype(int)
    

    continguous_dict = {}
    
    for new_id , id in enumerate(np.unique(data[:, 1])):
        data[data[:, 1] == id , 1] = new_id
        continguous_dict[new_id] = id
    
    frames = np.unique(data[:, 0])    
    matches = []

    for frame in frames:
        
        bboxes = data[data[:, 0] == frame]
     
        if len(bboxes) == 0:
            continue
       
        iou_scores = calculate_similarities(
            bboxes1=bboxes[:, 2:], bboxes2=bboxes[:, 2:], img_shape = img_shape)

        iou_scores-=np.eye(len(bboxes))
        row, col = np.where((iou_scores > 0))

        ioa_scores = calculate_similarities(
            bboxes1=bboxes[:, 2:], bboxes2=bboxes[:, 2:], img_shape = img_shape, do_ioa = True)
        
        ioa_scores-=np.eye(len(bboxes))
        


        # for r, c in zip(row, col):
        ids = np.array([continguous_dict[id] for id in bboxes[:, 1]]) 
        final_iou_scores = iou_scores[row, col]
        final_ioa_scores = ioa_scores[row, col]
        matches.append(np.vstack(((np.ones(len(row)) * frame).astype(int), ids[row], ids[col], final_iou_scores, final_ioa_scores)).T)
    df = pd.DataFrame(np.concatenate(matches), columns = ["frame", "id1", "id2", "IOU", "IOA"])
    df[["frame", "id1", "id2"]] = df[["frame", "id1", "id2"]].astype("int")
    return df


def overlap_frame(data_1, data_2,  img_shape = None):
    """ input data [frame, id, x, y,  w, t] """
    
    assert data_1.shape[-1] == 4, "data_1 wrong shape"
    assert data_2.shape[-1] == 4, "data_1 wrong shape"
   
    
    data_1 = data_1.astype(int)
    data_2 = data_2.astype(int)

  


        
    bboxes1 = data_1
    bboxes2 = data_2

    if len(bboxes2) == 0:
        return None
    if len(bboxes1) == 0:
        return None
  
    matches = []


    
    # iou_scores = calculate_similarities(
        # bboxes1=bboxes1[:, 2:], bboxes2=bboxes2[:, 2:], img_shape = img_shape)

    # iou_scores-=np.eye(len(bboxes1))
    # row, col = np.where((iou_scores > 0))

    ioa_scores = calculate_similarities(
        bboxes1=bboxes1, bboxes2=bboxes2, img_shape = img_shape, do_ioa = True)
    
    # ioa_scores-=np.eye(len(bboxes))
    
    

    # for r, c in zip(row, col):
    
    # final_iou_scores = iou_scores[row, col]

    final_ioa_scores_max = np.max((ioa_scores), 1)
    final_ioa_scores_min = np.min((ioa_scores), 1)

    return (final_ioa_scores_max > 0) * final_ioa_scores_max + (final_ioa_scores_max <= 0) * final_ioa_scores_min
    
    
 

def pix2real(H, pos,pixels, y0, img_width):
    x_pix = np.clip(pixels[:, 0], 0, img_width-1).astype(int)

    Ay = (H[1, 0] * pixels[:, 0] +  H[1, 1] * y0[x_pix] + H[1,2])
    Ax = (H[0, 0] * pixels[:,0] + H[0, 1] *  y0[x_pix] + H[0,2])
    B = (( H[2, 0]*pixels[:, 0] +   H[2, 1] * y0[x_pix] + H[2,2]))


    mask = pixels[:, 1] < y0[x_pix]
    converted_y =  (Ay/B - Ay/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix])) 
    converted_y[np.isnan(converted_y)] = 0

    converted_x = (Ax/B - Ax/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix]))
    converted_x[np.isnan(converted_x)] = 0
    pos[:,1 ] = pos[:, 1] * (1-mask)  + converted_y * mask
    pos[:,0 ] = pos[:, 0] * (1-mask)  + converted_x * mask

    return pos
def real2pix(H, pos, pixels, y0, img_width):
    x_pix = np.clip(pixels[:, 0], 0, img_width-1).astype(int)
    Ay = (H[1, 0] * pixels[:, 0] +  H[1, 1] * y0[x_pix] + H[1,2])

    B = (( H[2, 0]*pixels[:, 0]  + H[2, 1] * y0[x_pix] + H[2,2]))
    C = -H[0, 0] *H[1,1] * y0[x_pix]/H[1, 0] - H[0,0] * H[1,2]/H[1, 0] + H[0,1]* y0[x_pix] + H[0, 2]

    py = (pos[:, 0] - H[0, 0] * pos[:, 1] /H[1, 0]-(1/B + 1/B**2 *H[2, 1]* y0[x_pix]) * C)/(-1/B**2 * H[2, 1]* C)
    R = (1/B - 1/B**2 * H[2, 1]*(py - y0[x_pix]))
    px = (pos[:, 1] /R -H[1,1] * y0[x_pix]- H[1, 2])/H[1, 0]


    mask = pixels[:, 1] < y0[x_pix] 
    py = py[np.isnan(py)] = 0 
    px = px[np.isnan(px)] = 0 
    pixels[:,1 ] = pixels[:, 1] * (1-mask)  + py * mask
    pixels[:,0 ] = pixels[:, 0] * (1-mask)  + px * mask

    return pixels

if __name__ == "__main__":
    x = np.array([[781.3,  444.7 ,  72.3  ,207.5]])
    y  =np.array([ [808.79619393, 446.74116917 , 24.3   ,      80.        ]])
    score_mat = overlap_frame(y, x) 
        
    print(score_mat)
   