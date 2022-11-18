import os
from unicodedata import category
from attr import fields
import matplotlib.pyplot as plt

import numpy as np
from pycocotools import mask as coco_mask

import open3d as o3d
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import pandas as pd
from tqdm import tqdm
import math
from PIL import Image, ImageDraw
from numpy.linalg import inv
import traceback

import copy

# from dataset import PanopticLoader
from .utils import id_matching, id_matching_ioa 

# panoptic_loader = PanopticLoader()


def scale_intrinsics(intrinsics, old_shape, new_shape):
    """Scales intrinsics based on the image size change
    Args:
        intrinsics (ndarray): intrinsics matrix
        old_shape (tuple): old image dimensions  (hxw)
        new_shape (tuple): new image dimensions  (hxw)
    Returns:
        intrinsics (ndarray): rescaled intrinsics
    """
    # intrinsics = np.asarray(intrinsics.intrinsic_matrix, dtype=np.float64)
    # intrinsic_matrix[0, :] = intrinsic_matrix *new_shape[0] / old_shape[0]
    # intrinsic_matrix[1, :] = intrinsic_matrixnew_shape[1] / old_shape[1]

    intrinsic_matrix = copy.deepcopy(intrinsics.intrinsic_matrix)

    intrinsic_matrix[0, :] = intrinsic_matrix[0, :] * \
        new_shape[0] / old_shape[0]
    intrinsic_matrix[1, :] = intrinsic_matrix[1, :] * \
        new_shape[1] / old_shape[1]

    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        new_shape[0],
        new_shape[1],
        fx=intrinsic_matrix[0, 0],
        fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2],
        cy=intrinsic_matrix[1, 2],
    )

    return intrinsics


class DataSet:
    def __init__(self, prefix: str):
        self.prefix = prefix


class Sequence:
    def __init__(self, prefix, name, fields, dataset):
        self.prefix = prefix
        self.name = name
        self.matched_tracker = False
        self.matched_labels = False
        self.dataset = dataset
        # process sequence wide fields
        self.frames = []
        self.frame_cb = {}
        self.fields = fields
        self.pose_trajectory = None

        for field in fields.values():

            if field.setup_cb:
                data, frames = field.setup_cb(
                    os.path.join(prefix,field.location , name,  field.folder))

                # update frame information if possible

                if frames:

                    if len(frames) > len(self.frames):
                        self.frames = frames
                    elif len(frames) == len(self.frames):

                        assert self.frames == frames, "frames do not match for `{}`".format(
                            field.id)

                # if there's more info to store
                if data is not None:
                    setattr(self, field.id, data)
                # import pdb; pdb.set_trace()

            # print(field)
            if field.frame_cb:
                self.frame_cb[field.id] = field.frame_cb

        # import pdb; pdb.set_trace()
        # parse frames to process
        # print(f"Initialized sequence {name}")
    @property
    def has_egomotion(self):
       
        if not self.__getitem__(0, "egomotion")["egomotion"]:
            return False
        else: 
            return True

    def transform_lidar_cam2(self, key):
        lidar = self.__getitem__(key)["lidar"]
        calibration = self.__getitem__(key)["calibration"]
        img = self.__getitem__(key)["rgb"] / 255.

        velo = lidar.T
        camT = calibration["T_cam2_velo"] @ velo
        cam2 = camT[:3] / camT[2]
        uv = calibration["K_cam2"] @ cam2
        uv = uv[:2].astype(np.int32)  # [2, N]
        # mask out points outside the image
        mask_x = np.logical_and(uv[0] < img.shape[1], uv[0] >= 0)
        mask_y = np.logical_and(uv[1] < img.shape[0], uv[1] >= 0)

        mask = np.logical_and(mask_x, mask_y)
        mask = np.logical_and(mask, camT[2] > 0)
        uv = uv[:, mask]

        rgb = img[uv[1], uv[0]].T  # [3, N']
        camT = camT[:, mask]

        return camT,  rgb, uv, img

    def transform_depth_world(self, key, transform=True):
        item = self.__getitem__(key, ["depth",  "calibration", "rgb"])
        lidar = item["depth"]

        calibration = item["calibration"]
        img = item["rgb"]

        # img = np.array(Image.fromarray(img).resize(((960, 576)), Image.NEAREST))
        # calibration = scale_intrinsics(calibration, (1080, 1920), (576, 960))

        lidar = np.array(Image.fromarray(lidar).resize(
            ((1920, 1080)), Image.NEAREST))
        # calibration = scale_intrinsics(calibration, (1080, 1920), (576, 960))

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img),
            o3d.geometry.Image(lidar),
            convert_rgb_to_intensity=False,
            depth_scale=1,
            depth_trunc=1e10,
        )
        pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img, calibration)
        if transform:
            pose = self.__getitem__(key, ["pose"])["pose"]
            pt_cloud.transform(pose)

        return np.array(pt_cloud.points),  np.array(pt_cloud.colors), None, img, pt_cloud

    def transform_lidar_world(self, key, transform=True):
        item = self.__getitem__(key, ["depth", "pose", "calibration", "rgb"])
        depth = item["depth"]
        calibration = item["calibration"]
        img = item["rgb"]

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img),
            o3d.geometry.Image(depth),
            convert_rgb_to_intensity=False,
            depth_scale=1,
            depth_trunc=1e10,
        )
        pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img, calibration)

        if transform:
            pose = self.__getitem__(key, ["pose"])["pose"]
            pt_cloud.transform(pose)

        return np.array(pt_cloud.points),  np.array(pt_cloud.colors), None, img, pt_cloud

    def transform_lidar_world_mot(self, key):
        item = self.__getitem__(key, ["lidar", "pose", "calibration", "rgb"])
        lidar = item["lidar"]
        pose = item["pose"]
        intrinsics = item["calibration"]
        img = item["rgb"]

        import cv2

        lidar += np.finfo('float').eps
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img),
            o3d.geometry.Image(lidar),
            convert_rgb_to_intensity=False,
            depth_scale=1,
            depth_trunc=1e10,
        )

        height, width, _ = img.shape
        # intrinsics.set_intrinsics(width = width, height = height)

        pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img, intrinsics)

        pt_cloud.transform(pose)

        return np.array(pt_cloud.points),  np.array(pt_cloud.colors), None, img, pt_cloud

    def transform_lidar_world_kitti(self, key):
        item = self.__getitem__(
            key, ["panoptic", "lidar", "pose", "calibration", "rgb"])
        lidar = item["lidar"]
        calibration = item["calibration"]
        classes = item["panoptic"]["mask"]
        img = item["rgb"]
        pose = item["pose"]

        velo = lidar.T
        camT = calibration["T_cam2_velo"] @ velo
        cam2 = camT[:3] / camT[2]
        uv = calibration["K_cam2"] @ cam2
        uv = uv[:2].astype(np.int32)  # [2, N]
        # mask out points outside the image
        mask_x = np.logical_and(uv[0] < classes.shape[1], uv[0] >= 0)
        mask_y = np.logical_and(uv[1] < classes.shape[0], uv[1] >= 0)

        mask = np.logical_and(mask_x, mask_y)
        mask = np.logical_and(mask, camT[2] > 0)

        uv = uv[:, mask]

        rgb = img[uv[1], uv[0]].T  # [3, N']

        velo = velo[:, mask]

        velo = calibration["Tr_velo_imu"] @ velo
        velo = pose @ velo
        return velo, rgb, uv

    def get_pedestrian_pixels_lidar(self, key, mask=None):
        item = self.__getitem__(key, fields=["masks"])
        masks = item["masks"]
        pedestrian_dict = {}

        for index, row in masks.iterrows():
            id_pedestrian = int(row.id - 2000)
            segmentation = coco_mask.decode({"size": [row["height"], row["width"]],
                                             "counts": row["mask"]})
            # segmentation = np.array(Image.fromarray(mots_mask).resize(((960, 576)), Image.NEAREST))
            # if mask is not None:

            #     segmentation[mask[..., 0] == False] *= 0
            pixels = np.stack(
                np.where(segmentation == 1)).T
            if (len(pixels) == 0):
                continue
            pedestrian_dict[id_pedestrian] = pixels
        return pedestrian_dict

    def get_pedestrian_pixels(self, key, mask=None, scale=None):
        item = self.__getitem__(key, fields=["segmentation", "dets"])

        segmentation = item["segmentation"][:, :, :3]

        if mask is not None:
            segmentation[mask == False] *= 0
        dets = item["dets"]

        ids = dets.id.unique()

        pedestrian_dict = {}
        for id_pedestrian in ids:

            color = panoptic_loader.id_colors[id_pedestrian]

            pixels = np.stack(
                np.where(np.all(segmentation == np.array(color), axis=-1)), 1)

            pedestrian_dict[id_pedestrian] = pixels

        return pedestrian_dict

    def get_pedestrian_pixels_depth(self, key, mask=None):
        item = self.__getitem__(key, fields=["masks"])
        masks = item["masks"]
        pedestrian_dict = {}

        for index, row in masks.iterrows():
            id_pedestrian = int(row.id - 2000)
            mots_mask = coco_mask.decode({"size": [row["height"], row["width"]],
                                          "counts": row["mask"]})
            segmentation = np.array(Image.fromarray(
                mots_mask).resize(((960, 576)), Image.NEAREST))
            if mask is not None:

                segmentation[mask[..., 0] == False] *= 0
            pixels = np.stack(
                np.where(segmentation == 1)).T
            if (len(pixels) == 0):
                continue
            pedestrian_dict[id_pedestrian] = pixels
        return pedestrian_dict

    def get_pedestrian_positions_kitti(self, key):
        velo, rgb, uv = self.transform_lidar_world_kitti(key)
        item = self.__getitem__(key, fields=["segmentation", "dets", "pose"])
        pose = item["pose"]
        pose_position = -pose[:, -1:].T
        segmentation = item["segmentation"]
        dets = item["dets"]
        ids = dets.id.unique()

        pedestrian_pixels = segmentation[uv[1], uv[0]]

        mask = (pedestrian_pixels[:, -1] == 255)
        pedestrian_pixels = pedestrian_pixels[mask]

        veloPed = velo.T[mask]

        pedestrian_dict = {}
        for id_pedestrian in ids:

            color = panoptic_loader.id_colors[id_pedestrian]

            pixels = np.stack(
                np.where(np.all(pedestrian_pixels[:, :3] == np.array(color), axis=-1)), 1).reshape(-1)

            ped_points = veloPed[pixels]
            ped_points = (ped_points + pose_position)
            x, y = ped_points[:, 0], ped_points[:, 1]
            r = np.sqrt(x**2+y**2)
            t = np.arctan2(y, x)
            t_mean = np.mean(t)
            r_mean = np.quantile(r, .05)

            ped_position = np.array([r_mean * np.cos(t_mean), r_mean * np.sin(
                t_mean), np.min(ped_points[:, 2]), 0]) - pose_position[0]

            pedestrian_dict[id_pedestrian] = {"position": np.array(
                [ped_position[0], ped_position[2], ped_position[1],  0])}
        return pedestrian_dict

    def get_pedestrian_positions_mot(self, pedestrian_dict, points, colors, origin=np.array([0., 0.]),
                                     img_shape=(1080, 1920)):
        points_reshaped = np.reshape(points, (img_shape[0], img_shape[1], 3))
        colors_reshaped = np.reshape(colors, (img_shape[0], img_shape[1], 3))

        pedestrian = {}

        for id, pixels in pedestrian_dict.items():

            ped_points = points_reshaped[pixels[:, 0], pixels[:, 1], :]
            ped_color = colors_reshaped[pixels[:, 0], pixels[:, 1], :]
            x, y = ped_points[:, 0] - origin[0], ped_points[:, 2] - origin[1]
            r = np.sqrt(x**2+y**2)
            t = np.arctan2(y, x)
            t_mean = np.mean(t)
            r_mean = np.quantile(r, 0.5)

            try:
                height_mask = r <= np.quantile(r, .9)
                ground_position = np.min(ped_points[height_mask, 1])
            except:

                ground_position = np.min(ped_points[0, 1])

            points_list = [r_mean * np.cos(t_mean) + origin[0], r_mean * np.sin(
                t_mean) + origin[1], ground_position]
            for quantile in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                points_list.extend([
                    np.quantile(r, quantile) * np.cos(t_mean) + origin[0],
                    np.quantile(r, quantile) * np.sin(t_mean) + origin[1],
                    np.quantile(r, quantile)
                ])
            ped_position = np.array(points_list)

            pedestrian[id] = {"position": ped_position,
                              "points":  ped_points,
                              "pixel":   pixels,
                              "color": ped_color
                              }
        return pedestrian

    def get_pedestrian_positions(self, pedestrian_dict, points, img_shape=(1080, 1920)):
        points_reshaped = np.reshape(points, (img_shape[0], img_shape[1], 3))

        pedestrian = {}

        for id, pixels in pedestrian_dict.items():

            ped_points = points_reshaped[pixels[:, 0], pixels[:, 1], :]

            ped_position = np.median(ped_points, axis=0)

            pedestrian[id] = {"position": ped_position,
                              "points": ped_points,
                              "pixel": pixels}
        return pedestrian

    def match_tracker_world(self):
        assert "tracker" in self.__dict__.keys(), "`tracker` not set"
        assert "labels" in self.__dict__.keys(), "`labels` not set"

        self.tracker = self.tracker.merge(self.labels[["frame", "id", "x_world", "y_world", "z_world"]], left_on=[
                                          "frame", "gt_id"], right_on=["frame", "id"], how="left")
        self.tracker.drop(columns="id_y", inplace=True)
        self.tracker.rename(columns={"id_x": "id"}, inplace=True)

    def compute_world_labels(self):
        self.labels[["x_world", "y_world", "z_world"]] = None
        for frame in self.frames:
            pose = self.__getitem__(frame, "pose")["pose"]

            frame_mask = (self.labels.frame == frame)
            coordinates = self.labels[frame_mask][["x", "z", "y"]].values / 0.9

            coordinates_transformed = (
                pose[:3, :3] @ coordinates.T).T + (pose[:3, -1:].T)

            self.labels.loc[frame_mask,
                            "x_world"] = coordinates_transformed[:, 0]
            self.labels.loc[frame_mask,
                            "y_world"] = coordinates_transformed[:, 2]
            self.labels.loc[frame_mask,
                            "z_world"] = coordinates_transformed[:, 1]

    def create_scene_data(self, scale=1, height_threshold=None, frames=None):

        # Helper functions to convert polar <=> cartesian coordinates

        def cart2pol(x, y):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return (rho, phi)

        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        df = pd.DataFrame(columns=["x", "y", "class", "r", "g", "b"])
        visibility_map = {}
        if frames is None:
            frames = np.arange(1, self.__len__()+1)

        for index, frame in enumerate(tqdm(frames)):
            frame_df = pd.DataFrame(columns=["x", "y", "class"])
            # get all data needed
            item = self.__getitem__(
                frame, ["panoptic", "lidar", "pose", "calibration", "rgb", "labels"])
            classes = item["panoptic"]["mask"]

            lidar = item["lidar"]
            pose = item["pose"]
            calibration = item["calibration"]
            img = item["rgb"]
            labels = item["labels"]

            coordinates = labels[["x", "z", "y"]].values
            flat_classes = np.reshape(classes, -1)

            #  laebl coordinates => world coordinates
            label_cloud = o3d.geometry.PointCloud()
            label_cloud.points = o3d.utility.Vector3dVector(coordinates)

            label_cloud.transform(pose)

            # compute position ground plane

            ground_offset = np.mean(np.array(label_cloud.points)[:, 1])

            # create point cloud from image and depth
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(img),
                o3d.geometry.Image(lidar),
                convert_rgb_to_intensity=False,
                depth_scale=1,
                depth_trunc=1e10,
            )
            pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img, calibration)

            # filter furtherst points
            mask = [np.array(pt_cloud.points)[:, -1] <
                    (np.max(np.array(pt_cloud.points)[:, -1]) - 0.2)]

            points = np.array(pt_cloud.points)[mask]
            flat_classes = flat_classes[mask]

            pt_cloud.points = o3d.utility.Vector3dVector(points)
            pt_cloud.colors = o3d.utility.Vector3dVector(
                np.array(pt_cloud.colors)[mask])
            # transform pt cloud image => world coordinates
            pt_cloud.transform(pose)

            mask_map = [
                (abs(np.array(pt_cloud.points)[:, 1] - ground_offset) < 3)]
            camera_position = np.array([pose[0, 3], pose[2, 3]])
            pt_cloud.points = o3d.utility.Vector3dVector(
                (np.array(pt_cloud.points)[mask_map]))
            pt_cloud.colors = o3d.utility.Vector3dVector(
                (np.array(pt_cloud.colors)[mask_map]))
            flat_classes = flat_classes[mask_map]
            mask_vis = [
                (abs(np.array(pt_cloud.points)[:, 1] - ground_offset) < 1.8)]
            points_vis = np.array(pt_cloud.points)[mask_vis]
            points = np.array(pt_cloud.points)

            # visibility map, world => polar coordinates
            rho, phi = cart2pol(
                points_vis[:, 0] - camera_position[0], points_vis[:, 2] - camera_position[1])
            polar_df = pd.DataFrame({"rho": rho, "phi": phi})

            polar_df["phi"] = polar_df["phi"].round(2)
            max_rho = polar_df.groupby("phi")["rho"].max().reset_index()
            max_rho.sort_values("phi", inplace=True)

            # print(max_rho.describe())
            polar_coordinates = max_rho[["rho", "phi"]].values
            rho = polar_coordinates[:, 0]
            phi = polar_coordinates[:, 1]
            (x, y) = pol2cart(rho, phi)

            # compute origin
            # origin = [np.min(x) , np.min(y)]
            # x-= np.min(x)
            # y-= np.min(y)
            cartesian_df = pd.DataFrame(
                {"x": x + camera_position[0], "y": y + camera_position[1]})

            cartesian_xy = cartesian_df[["x", "y"]].values
            label_cloud = o3d.geometry.PointCloud()

            polygon = [(camera_position[0], camera_position[1])] + \
                [(p[0], p[1]) for p in cartesian_xy]

            # img = Image.new('L', (width, height), 0)

            # ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            # polygon_mask = np.array(img)
            visibility_map_t = {}
            # visibility_map_t = coco_mask.encode(np.asfortranarray(polygon_mask))
            visibility_map_t["offset"] = list(camera_position)
            visibility_map_t["polygon"] = polygon

            visibility_map[str(frame)] = visibility_map_t
            points = np.array(pt_cloud.points).T

            rgb = np.array(pt_cloud.colors).T*255

            sky_mask = flat_classes != 0

            frame_df["x"] = points[0][sky_mask]
            frame_df["y"] = points[2][sky_mask]
            frame_df["class"] = flat_classes[sky_mask]
            frame_df["r"] = rgb[0][sky_mask]
            frame_df["g"] = rgb[1][sky_mask]
            frame_df["b"] = rgb[2][sky_mask]

            frame_df["y"] = (frame_df["y"]/scale).astype("int32")*scale
            frame_df["x"] = (frame_df["x"]/scale).astype("int32")*scale
            df = pd.concat([df, frame_df])

            if ((index % 5 == 0) or (index == frames[-1])):
                df.drop_duplicates(subset=["x", "y"], inplace=True)
                idx = df.groupby(["x", "y"])['class'].transform(
                    max) == df['class']

                df = df[idx]

            # o3d.visualization.webrtc_server.enable_webrtc()
            # o3d.visualization.draw([label_cloud, pt_cloud] )
        df[["r", "g", "b"]] = df[["r", "g", "b"]].astype("int32")
        keys = list(visibility_map.keys())
        for t in keys:
            points = visibility_map[t]["polygon"]
            origin = np.array(visibility_map[t]["offset"])
            p = np.array(points)
            # print(origin, p)

            d = p - origin
            phi = np.arctan2(d[:, 1], d[:, 0])

            index = list(np.argsort(phi))

            visibility_map[t]["polygon"] = [points[i] for i in index]

        return df, visibility_map

    def get_lidar_gradient_mask(self, frame, gradient_threshold):

        lidar = self.__getitem__(frame, "lidar")["lidar"]
        import cv2

        def get_gradient(depth):
            sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
            gradient = abs(sobelx + sobely)
            return gradient
        gradient = get_gradient(lidar)

        gradient_mask = gradient < gradient_threshold
        gradient_mask = np.tile(gradient_mask[..., np.newaxis], (1, 1, 3))
        return gradient_mask

    def get_depth_gradient_mask(self, frame, gradient_threshold):

        depth = self.__getitem__(frame, "depth")["depth"]
        import cv2

        def get_gradient(depth):
            sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
            gradient = abs(sobelx + sobely)
            return gradient
        gradient = get_gradient(depth)

        gradient_mask = gradient < gradient_threshold
        gradient_mask = np.tile(gradient_mask[..., np.newaxis], (1, 1, 3))
        return gradient_mask

    def create_scene_data_kitti(self, scale=1, height_threshold=None, frames=None):
        df = pd.DataFrame(columns=["x", "y", "class", "r", "g", "b"])

        if frames is None:
            frames = np.arange(0, self.__len__())
        for index, frame in enumerate(tqdm(frames)):

            frame_df = pd.DataFrame(columns=["x", "y", "class"])
            # get all data needed
            item = self.__getitem__(
                frame, ["panoptic", "lidar", "pose", "calibration", "rgb"])
            lidar = item["lidar"]

            calibration = item["calibration"]
            classes = item["panoptic"]["mask"]
            img = item["rgb"]
            pose = item["pose"]

            velo = lidar.T
            camT = calibration["T_cam2_velo"] @ velo
            cam2 = camT[:3] / camT[2]
            uv = calibration["K_cam2"] @ cam2
            uv = uv[:2].astype(np.int32)  # [2, N]
            # mask out points outside the image
            mask_x = np.logical_and(uv[0] < classes.shape[1], uv[0] >= 0)
            mask_y = np.logical_and(uv[1] < classes.shape[0], uv[1] >= 0)

            mask = np.logical_and(mask_x, mask_y)
            mask = np.logical_and(mask, camT[2] > 0)

            uv = uv[:, mask]
            classes = classes[uv[1], uv[0]].T  # [3, N']

            rgb = img[uv[1], uv[0]].T  # [3, N']

            velo = velo[:, mask]

            velo = calibration["Tr_velo_imu"] @ velo
            velo = pose @ velo

            sky_mask = classes != 0

            velo[2] = 0
            frame_df["x"] = velo[0][sky_mask]
            frame_df["y"] = velo[1][sky_mask]
            frame_df["class"] = classes[sky_mask]
            frame_df["r"] = rgb[0][sky_mask]
            frame_df["g"] = rgb[1][sky_mask]
            frame_df["b"] = rgb[2][sky_mask]

            frame_df["y"] = (frame_df["y"]/scale).astype("int32")*scale
            frame_df["x"] = (frame_df["x"]/scale).astype("int32")*scale
            df = pd.concat([df, frame_df])

            if ((index % 5 == 0) or (index == frames[-1])):
                df.drop_duplicates(subset=["x", "y"], inplace=True)
                idx = df.groupby(["x", "y"])['class'].transform(
                    max) == df['class']

                df = df[idx]

        df = df[idx]
        return df

    def create_scene_data_mot(self, scale=1, height_threshold=None, frames=None):

        # Helper functions to convert polar <=> cartesian coordinates

        def cart2pol(x, y):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return (rho, phi)

        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        df = pd.DataFrame(columns=["x", "y", "class", "r", "g", "b"])
        visibility_map = {}
        if frames is None:
            frames = np.arange(1, self.__len__()+1)

        for index, frame in enumerate(tqdm(frames)):
            frame_df = pd.DataFrame(columns=["x", "y", "class"])
            # get all data needed
            item = self.__getitem__(
                frame, ["panoptic", "lidar", "pose", "calibration", "rgb"])
            classes = item["panoptic"]["mask"]
            flat_classes = np.reshape(classes, -1)
            lidar = np.maximum(item["lidar"], 0)
            lidar += np.finfo('float').eps
            pose = item["pose"]
            intrinsics = item["calibration"]
            img = item["rgb"]

            import cv2

            gradient_threshold = 50
            gradient_mask = self.get_depth_gradient_mask(
                frame, gradient_threshold)

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(img),
                o3d.geometry.Image(lidar),
                convert_rgb_to_intensity=False,
                depth_scale=1,
                depth_trunc=1e10,
            )

            height, width, _ = img.shape

            # intrinsics.set_intrinsics(width = width, height = height)

            pt_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img, intrinsics)
            pt_cloud.transform(pose)
            points = np.array(pt_cloud.points)
            colors = np.array(pt_cloud.colors)

            min_y = np.min(points[:, 1])

            # filter furtherst points
            mask = np.array(pt_cloud.points)[
                :, -1] < (np.max(np.array(pt_cloud.points)[:, -1]) - 0.2)

            # # filter gradient threshold

            mask = np.logical_and(
                mask, np.reshape(gradient_mask[:, :, 0], -1))
            # mask = np.logical_and(mask, (points[:, 1] < (min_y + 5)))

            # mask = [True] * len(points)
            points = points[mask]
            colors = colors[mask]

            flat_classes = flat_classes[mask]

            # pt_cloud.points = o3d.utility.Vector3dVector(points)
            # pt_cloud.colors = o3d.utility.Vector3dVector(
            #     np.array(pt_cloud.colors)[mask])
            # # transform pt cloud image => world coordinates
            # pt_cloud.transform(pose)

            camera_position = np.array([0., 0.])
            # pt_cloud.points = o3d.utility.Vector3dVector((np.array(pt_cloud.points)[mask_map]))
            # pt_cloud.colors = o3d.utility.Vector3dVector((np.array(pt_cloud.colors)[mask_map]))
            # flat_classes = flat_classes[mask_map]
            # mask_vis = [( abs(np.array(pt_cloud.points)[:, 1] - ground_offset)  <  1.8 )]
            # points_vis = np.array(pt_cloud.points)[mask_vis]
            # points = np.array(pt_cloud.points)

            # visibility map, world => polar coordinates
            rho, phi = cart2pol(
                points[:, 0] - camera_position[0], points[:, 2] - camera_position[1])
            polar_df = pd.DataFrame({"rho": rho, "phi": phi})

            polar_df["phi"] = polar_df["phi"].round(2)
            max_rho = polar_df.groupby("phi")["rho"].max().reset_index()
            max_rho.sort_values("phi", inplace=True)

            # print(max_rho.describe())
            polar_coordinates = max_rho[["rho", "phi"]].values
            rho = polar_coordinates[:, 0]
            phi = polar_coordinates[:, 1]
            (x, y) = pol2cart(rho, phi)

            # compute origin
            # origin = [np.min(x) , np.min(y)]
            # x-= np.min(x)
            # y-= np.min(y)
            cartesian_df = pd.DataFrame(
                {"x": x + camera_position[0], "y": y + camera_position[1]})

            cartesian_xy = cartesian_df[["x", "y"]].values
            # label_cloud = o3d.geometry.PointCloud()

            polygon = [(camera_position[0], camera_position[1])] + \
                [(p[0], p[1]) for p in cartesian_xy]

            # img = Image.new('L', (width, height), 0)

            # ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            # polygon_mask = np.array(img)
            visibility_map_t = {}
            # # visibility_map_t = coco_mask.encode(np.asfortranarray(polygon_mask))
            visibility_map_t["offset"] = list(camera_position)
            visibility_map_t["polygon"] = polygon

            visibility_map[str(frame)] = visibility_map_t
            points = points.T

            rgb = colors.T*255

            sky_mask = flat_classes != 0

            frame_df["x"] = points[0][sky_mask]
            frame_df["y"] = points[2][sky_mask]
            frame_df["class"] = flat_classes[sky_mask]
            frame_df["r"] = rgb[0][sky_mask]
            frame_df["g"] = rgb[1][sky_mask]
            frame_df["b"] = rgb[2][sky_mask]

            frame_df["y"] = (frame_df["y"]/scale).astype("int32")*scale
            frame_df["x"] = (frame_df["x"]/scale).astype("int32")*scale
            df = pd.concat([df, frame_df])

            if ((index % 5 == 0) or (index == frames[-1])):
                df.drop_duplicates(subset=["x", "y"], inplace=True)
                idx = df.groupby(["x", "y"])['class'].transform(
                    max) == df['class']

                df = df[idx]

            # o3d.visualization.webrtc_server.enable_webrtc()
            # o3d.visualization.draw([label_cloud, pt_cloud] )
        df[["r", "g", "b"]] = df[["r", "g", "b"]].astype("int32")
        keys = list(visibility_map.keys())
        for t in keys:
            points = visibility_map[t]["polygon"]
            origin = np.array(visibility_map[t]["offset"])
            p = np.array(points)
            # print(origin, p)

            d = p - origin
            phi = np.arctan2(d[:, 1], d[:, 0])

            index = list(np.argsort(phi))

            visibility_map[t]["polygon"] = [points[i] for i in index]

        return df, visibility_map

    def match_dets(self, right="tracker", positions=False, ioa=True):
        if right == "tracker" and self.matched_tracker:
            return

        elif right == "labels" and self.matched_labels:
            return

        def match(left_data, right_data, height=1080, width=1920, ioa=True):
            matches, matches_list = id_matching(
                left_data, right_data, img_shape=(height, width))
            # matches_list = []
            if ioa:
                right_residual = right_data
                left_residual = left_data

                for match in matches_list:

                    index = np.where((right_residual[:, 0] == match[0]) & (
                        right_residual[:, 1] == match[1]))

                    right_residual = np.delete(
                        right_residual, (index[0][0]), axis=0)
                    index = np.where((left_residual[:, 0] == match[0]) & (
                        left_residual[:, 1] == match[2]))
                    left_residual = np.delete(
                        left_residual, (index[0][0]), axis=0)

                matches, matches_list_ioa = id_matching_ioa(
                    left_residual, right_residual, img_shape=(height, width))

                matches_list.extend(matches_list_ioa)
            return matches_list

        img = self.__getitem__(1, fields=["rgb"])["rgb"]
        height, width, _ = img.shape
        left_df = self.dets
        assert "dets" in self.__dict__.keys(), "`dets` not set"
        if right == "tracker":
            assert "tracker" in self.__dict__.keys(), "`tracker` not set"
            right_df = self.tracker
            right_model_df = right_df.groupby(["tracker", "model"])[
                "frame"].count().reset_index()

        elif right == "labels":
            assert "labels" in self.__dict__.keys(), "`labels` not set"
            right_df = self.labels
            right_data = right_df[["frame", "id", "bb_left",
                                   "bb_top", "bb_width", "bb_height"]].values

        left_data = left_df[["frame", "id", "bb_left",
                             "bb_top", "bb_width", "bb_height"]].values
        df_list = []

        if right == 'tracker':
            for index, row in right_model_df.iterrows():
                # account for (out of window bounding boxes)

                right_data = right_df[((right_df.tracker == row.tracker) & (right_df.model == row.model))][[
                    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]].values

                matches_list = match(
                    left_data, right_data, height=height, width=width, ioa=True)

                df = pd.DataFrame(matches_list,  columns=[
                    "frame", "id_tracker", "id_dets"])
                df["tracker"] = row.tracker
                df["model"] = row.model

                df_list.append(df)

            df_out = pd.concat(df_list)

            right_df = right_df.merge(df_out, left_on=["tracker", "model", "frame", "id"],
                                      right_on=["tracker", "model", "frame", "id_tracker"], how="left")
        elif right == "labels":
            matches_list = match(
                left_data, right_data, height=height, width=width, ioa=True)
            df_out = pd.DataFrame(matches_list,  columns=[
                "frame", "id_gt", "id_dets"])

            right_df = right_df.merge(df_out, left_on=["frame", "id"],
                                      right_on=["frame", "id_gt"], how="left")

        if positions:
            assert "positions" in self.__dict__.keys(), "`positions` not set"
            positions_df = self.positions
            positions_df.rename(columns={"id": "id_dets"}, inplace=True)

            right_df = right_df.merge(positions_df, left_on=["id_dets", "frame"],
                                      right_on=["id_dets", "frame"], how="left")

        if right == "tracker":
            self.tracker = right_df
            self.tracker[["id_tracker", "id_dets"]] = self.tracker[[
                "id_tracker", "id_dets"]].fillna(value=-1)
            self.tracker[["id_tracker", "id_dets"]] = self.tracker[[
                "id_tracker", "id_dets"]].astype("int")
            self.matched_tracker = True
        elif right == "labels":
            self.labels = right_df
            self.labels[["id_gt", "id_dets"]] = self.labels[[
                "id_gt", "id_dets"]].fillna(value=-1)
            self.labels[["id_gt", "id_dets"]] = self.labels[[
                "id_gt", "id_dets"]].astype("int")

            self.matched_labels = True

    def world_to_pose(self, frame, X):
        pose = self.__getitem__(frame, "pose")["pose"]
        inv_pose = inv(pose)

        return (inv_pose[:3, :3] @ X.T).T + inv_pose[:3, -1:].T

    def project_3d_to_2d(self, x, y, z):
        calibration = self.__getitem__(1, "calibration")["calibration"]
        fx, fy = calibration.get_focal_length()
        (cx, cy) = calibration.get_principal_point()

        u = fx * x / z + cx
        v = fy * y / z + cy
        return u, v

    def project_homography(self, x, y, frame, homography="homography_depth", y0=None):

        def get_y0(H, img_width):

            x_array = np.arange(0, img_width)
            horizon = -(H[2, 0] * x_array + H[2, 2]) / H[2, 1]
            y0_list = []
            for h, x in zip(horizon, x_array):

                y = np.arange(np.ceil(h)+1, 1080)

                xx = np.ones(len(y)) * x
                p = np.stack((xx, y, np.ones(len(y))))
                pp = H.dot(p).T
                pp = pp[:, :2]/pp[:, -1:]
                dd = pp[1:, 1] - pp[:-1, 1]

                dk = dd[1:] / dd[:-1]

                pix_y = y[1:]
                lower_threshold = pix_y[abs(dd) > .2]

                if len(lower_threshold) == 0:
                    y0_list.append(h + 40)
                else:
                    y0_list.append(lower_threshold[-1])
            return np.array(y0_list)

        def pix2real(H, pos, pixels, y0, img_width):
            x_pix = np.clip(pixels[:, 0], 0, img_width-1).astype(int)

            Ay = (H[1, 0] * pixels[:, 0] + H[1, 1] * y0[x_pix] + H[1, 2])
            Ax = (H[0, 0] * pixels[:, 0] + H[0, 1] * y0[x_pix] + H[0, 2])
            B = ((H[2, 0]*pixels[:, 0] + H[2, 1] * y0[x_pix] + H[2, 2]))

            mask = pixels[:, 1] < y0[x_pix]
            converted_y = (Ay/B - Ay/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix]))
            converted_y[np.isnan(converted_y)] = 0

            converted_x = (Ax/B - Ax/B**2 * H[2, 1]*(pixels[:, 1] - y0[x_pix]))
            converted_x[np.isnan(converted_x)] = 0
            pos[:, 1] = pos[:, 1] * (1-mask) + converted_y * mask
            pos[:, 0] = pos[:, 0] * (1-mask) + converted_x * mask

            return pos

        def real2pix(H, pos, pixels, y0, img_width):

            x_pix = np.clip(pixels[:, 0], 0, img_width-1).astype(int)
            # print(x_pix.shape, x_pix.min(), x_pix.max(), pixels[:, 0].min(), pixels[:, 0].max())
            # print(y0.shape)
            # print(pixels.shape)

            Ay = (H[1, 0] * pixels[:, 0] + H[1, 1] * y0[x_pix] + H[1, 2])

            B = ((H[2, 0]*pixels[:, 0] + H[2, 1] * y0[x_pix] + H[2, 2]))
            C = -H[0, 0] * H[1, 1] * y0[x_pix]/H[1, 0] - H[0, 0] * \
                H[1, 2]/H[1, 0] + H[0, 1] * y0[x_pix] + H[0, 2]

            py = (pos[:, 0] - H[0, 0] * pos[:, 1] / H[1, 0]-(1/B + 1 /
                  B**2 * H[2, 1] * y0[x_pix]) * C)/(-1/B**2 * H[2, 1] * C)
            R = (1/B - 1/B**2 * H[2, 1]*(py - y0[x_pix]))
            px = (pos[:, 1] / R - H[1, 1] * y0[x_pix] - H[1, 2])/H[1, 0]

            mask = pixels[:, 1] < y0[x_pix]
            py[np.isnan(py)] = 0
            px[np.isnan(px)] = 0
            pixels[:, 1] = pixels[:, 1] * (1-mask) + py * mask
            pixels[:, 0] = pixels[:, 0] * (1-mask) + px * mask

            return pixels

        H_dict = self.__getitem__(frame, [homography])[homography]

        try:
            egomotion = self.__getitem__(frame, ["egomotion"])["egomotion"]

        except:

            egomotion = None
        img = self.__getitem__(frame, ["rgb"])["rgb"]
        height, width, _ = img.shape
        inv_IPM = np.array(H_dict["inv_IPM"])
        assert len(x) == len(y)
        pos = np.zeros((len(x), 4))
        pos[:, (0)] = x
        pos[:, (2)] = y

        if egomotion is not None:
            H = np.array(H_dict["IPM"])
            y0 = get_y0(H, width)

            offset = np.array(H).dot(np.array([[int(width/2), height, 1]]).T).T
            offset = offset/offset[:, -1:]

            pos[:, (0, 2)] = pos[:, (0, 2)] + \
                egomotion["median"][np.newaxis, :2] + offset[:, :2]

        pos[:,  -1] = 1

        pos = pos[:, (0, 2, 3)]
        pixel = inv_IPM.dot(pos.T)

        pixel = pixel[:2]/pixel[-1:]
        pixel = pixel.T
        pixel[np.isnan(pixel[:, 0]), 0] = 0
        pixel[np.isnan(pixel[:, 1]), 1] = 3000
        pos[np.isnan(pixel[:, 0]), 0] = 0
        pos[np.isnan(pixel[:, 1]), 1] = 0
        if y0 is not None:
            try:

                pixel = real2pix(
                    np.array(H_dict["IPM"]), pos[:, (0, 1)], pixel, y0, width)
            except:
                print(traceback.print_exc())
                print(y0)
                print(y0.shape)
                print(width)
                dsd
        return pixel[:, 0], pixel[:, 1]
    # PLOT FUNCTIONS

    @property
    def funcs_plot(self):
        method_list = [func for func in dir(self)]
        return [func for func in method_list if "plot_" in func]

    def plot_positions(self,
                       frame=0,
                       ax=None,
                       plot_scene=True,
                       scene_type="coco",
                       fields=["positions"],
                       limit=True,
                       show=True,
                       xlim=None,
                       ylim=None,
                       length=1,
                       legend=False,
                       id_matched=True,
                       tracker="center_track",
                       model="motSynth_fulltest_private_motsynth",
                       boundary=25):

        scene_rgb, scene_category, metadata, visibility = self.__getitem__(
            1, fields=["map"])["map"].values()

        assert scene_type in [
            "coco", "category"], "Scene type `{}` not valid; Choose `coco` or `category`".format(scene_type)
        for c in fields:
            assert c in ["positions", "tracker",
                         "labels"], "class `{}` not valid".format(c)

        df_positions = []
        for f in np.arange(frame, frame + length):
            if f > self.__len__():
                continue
            item = self.__getitem__(f, fields=fields + ["pose"])
            if "positions" in fields:
                positions = item["positions"]

                positions["label"] = "DETS"
                df_positions.append(
                    positions[["frame", "id", "x", "y", "z", "label"]])

            if "labels" in fields:
                labels = item["labels"]

                pose = item["pose"]
                coordinates = labels[["x", "z", "y"]].values * 1 / 0.9

                coordinates_transformed = (
                    pose[:3, :3] @ coordinates.T).T + (pose[:3, -1:].T)

                labels["x"] = coordinates_transformed[:, 0]
                labels["y"] = coordinates_transformed[:, 2]
                labels["z"] = coordinates_transformed[:, 1]

                labels["label"] = "GT"
                df_positions.append(
                    labels[["frame", "id", "x", "y", "z", "label"]])

            if "tracker" in fields:
                if id_matching:
                    self.match_dets(right="Tracker", positions=True)
                tracker_df = item["tracker"]
                tracker_df = tracker_df[tracker_df.id_tracker > -1]
                tracker_df = tracker_df[(
                    (tracker_df.tracker == tracker) & (tracker_df.model == model))]
                tracker_df["label"] = "TRACKER"
                df_positions.append(
                    labels[["frame", "id", "x", "y", "z", "label"]])

        assert len(df_positions), "No data to plot"

        df_positions = pd.concat(df_positions)

        if scene_type == "coco":
            scene_img = scene_rgb
        else:
            scene_img = scene_category
            print(scene_category)
        if ax is None:
            fig, ax = plt.subplots()
        # if plot_scene:

        #     ax.imshow(scene_img)
        x_min, x_max, y_min, y_max = None, None, None, None

        for index, row in df_positions.iterrows():

            pos_x = (row["x"] - metadata["x_min"]) * 1/metadata["scale"]
            pos_y = (row["y"] - metadata["y_min"]) * 1/metadata["scale"]

            if row.label == "GT":
                marker = "^"
            elif row.label == "DETS":
                marker = "."
            ax.scatter(pos_x, pos_y, s=500, edgecolors='black', color=np.array(
                [*panoptic_loader.id_colors[int(row.id)], 255])/255., marker=marker,
                label=row.label, alpha=((row.frame - frame) / length))

            if x_min is not None:
                x_min = np.minimum(x_min, pos_x - 10)
            else:
                x_min = pos_x - boundary
            if x_max is not None:
                x_max = np.maximum(x_max, pos_x + 10)
            else:
                x_max = pos_x + boundary
            if y_min is not None:
                y_min = np.minimum(y_min, pos_y - 10)
            else:
                y_min = pos_y - boundary
            if y_max is not None:
                y_max = np.maximum(y_max, pos_y + 10)
            else:
                y_max = pos_y + boundary
        if limit:
            ax.set_xlim(x_max, x_min)
            ax.set_ylim(y_max, y_min)
        if legend:
            ax.legend()
        if show:
            plt.show()
        return ax

    def plot_set(self, frame, command=["rgb"], ax=None, show=True):
        available_commands = [func.replace(
            "plot_", "") for func in self.funcs_plot]

        for c in command:
            assert c in available_commands, "Invalid command: `{}`; Supported commands `{}`".format(
                c, "`, `".join(available_commands))
        ax = getattr(self, "plot_{}".format(command[0]))(
            frame,  show=False, ax=ax)

        for c in command[1:-1]:

            ax = getattr(self, "plot_{}".format(c))(frame, ax=ax, show=False)

        ax = getattr(self, "plot_{}".format(
            command[-1]))(frame, ax=ax, show=show)
        return ax

    def plot_visibility(self, key, scene_image=None, ax=None, show=True):

        scene_rgb, scene_category, metadata, visibility = self.__getitem__(key,  ["map"])[
            "map"].values()
        min_values = np.array([metadata["x_min"], metadata["y_min"]])

        origin = - min_values  # - np.array(visibility["offset"])
        scale = metadata["scale"]

        # print(visibility["polygon"])
        if ax is None:
            fig, ax = plt.subplots()

        if scene_image:
            plt.imshow(scene_rgb)
        p = Polygon([[(p[0] + origin[0])/scale, (p[1] + origin[1])/scale]
                    for p in visibility["polygon"]],  facecolor='green', alpha=0.3)
        ax.add_patch(p)

        if show:
            plt.show()
        return ax

    def plot_scene(self, df, scale=0.25):
        min_values = df[["x", "y"]].min().values

        df[["x_norm", "y_norm"]] = df[["x", "y"]] - df[["x", "y"]].min()
        img_rgb = np.zeros(
            (int(df["y_norm"].max()*(1/scale)) + 1, int(df["x_norm"].max()*(1/scale)) + 1, 4))
        img_class = np.zeros(
            (int(df["y_norm"].max()*(1/scale)) + 1, int(df["x_norm"].max()*(1/scale)) + 1))

        df["pixel_x"] = (df["x_norm"]*(1/scale)).astype("int32")
        df["pixel_y"] = (df["y_norm"]*(1/scale)).astype("int32")

        img_rgb[df["pixel_y"].values, df["pixel_x"].values,
                :3] = df[["r", "g", "b"]].values/255.
        img_rgb[df["pixel_y"].values, df["pixel_x"].values, -1] = 1
        img_class[df["pixel_y"].values, df["pixel_x"].values] = df["class"]
        img_class[df["pixel_y"].values, df["pixel_x"]] = img_class[df["pixel_y"].values,
                                                                   df["pixel_x"]] / len(panoptic_loader.categories)
        return img_rgb, img_class, (min_values)

    def plot_scene_img(self, ax=None, show=True):

        img_rgb, img_classes, metadata = self.__getitem__(0)["map"].values()

        if ax is None:
            ax = plt.gca()
        ax.imshow(img_rgb)
        if show:
            plt.show()
        return ax

    def plot_depth_kitti(self, key, ax=None, show=True):
        camT, rgb, uv, img = self.transform_lidar_cam2(key)
        depth_image = np.ones((img.shape[0], img.shape[1], 4))

        camT_norm = camT[2, :] / np.max(camT[2, :])
        for u, v, z in zip(uv[0], uv[1], camT_norm):

            depth_image[v, u] = np.array([1.-z, z, 0.5*z, z])
        if ax is None:
            ax = plt.gca()
        ax.imshow(depth_image, alpha=0.8)
        if show:
            plt.show()
        return ax

    def plot_rgb(self,  key, ax=None, show=True):
        item = self.__getitem__(key, fields=["rgb"])
        rgb = item["rgb"]
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb)
        if show:
            plt.show()
        return ax

    def plot_depth_motsynth(self,  key, ax=None, show=True):
        item = self.__getitem__(key)
        rgb = item["depth"]
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb)
        if show:
            plt.show()
        return ax

    def plot_depth(self,  key, ax=None, show=True):
        return self.plot_depth_motsynth(key=key,  ax=ax, show=show)

    def plot_panoptic(self,  key, style="coco", ax=None, show=True):
        item = self.__getitem__(key)
        assert style in [
            "coco", "category"], "`style`:{} not valid, Choose `coco` or `category`".format(style)
        rgb = item["panoptic"][style]
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb, alpha=0.5)
        if show:
            plt.show()
        return ax

    def plot_dets(self,  key, ax=None, show=True):
        item = self.__getitem__(key)
        dets = item["dets"]

        if ax is None:
            ax = plt.gca()
        for index, row in dets.iterrows():

            p = row[["bb_left", "bb_top", "bb_width", "bb_height"]].values

            rect = patches.Rectangle((p[0], p[1]), p[2], p[3], linewidth=1, edgecolor=np.array(
                [*panoptic_loader.id_colors[row.id], 255])/255., facecolor='none')
            ax.add_patch(rect)

        if show:
            plt.show()
        return ax

    def plot_tracker(self,
                     key,
                     tracker="center_track",
                     model="motSynth_fulltest_private_motsynth",
                     score=None,
                     tracker_ids=None,
                     max_age=None,
                     id_matching=True,
                     show_evaluation=False,
                     ax=None, show=True):
        if id_matching:
            self.match_dets(right="tracker")

        item = self.__getitem__(key, ["tracker", "labels"])

        df = item["tracker"]
        if tracker is not None:
            df = df[(df.tracker == tracker)]
        if model is not None:
            df = df[(df.model == model)]

        # df_false_negative = df_labels[(df_labels.id.isin(df.gt_id.unique()) == False)]

        if score is not None:
            df = df[df.score > score]

        if max_age is not None:
            df = df[df.age <= max_age]

        if ax is None:
            ax = plt.gca()

        if tracker_ids is not None:
            df = df[df.id.isin(tracker_ids)]

            assert len(df) > 0, "IDs `{}` not in sequence".format(
                ", ".join([str(t) for t in tracker_ids]))

        for index, row in df.iterrows():

            p = row[["bb_left", "bb_top", "bb_width", "bb_height"]].values

            if id_matching:

                if row.id_dets > -1:

                    color_id = row.id_dets
                    color = np.array(
                        [*panoptic_loader.id_colors[color_id], 255])/255.
                else:
                    color = np.ones(4)
            else:

                color = np.array(
                    [*panoptic_loader.id_colors[row.id], 255])/255.
            if show_evaluation:

                if math.isnan(row.gt_id):
                    facecolor = "red"

                else:
                    facecolor = "green"
                    if row.idsw == 1:
                        facecolor = "blue"

                rect = patches.Rectangle(
                    (p[0]+1, p[1]-1), p[2]-1, p[3]-1, alpha=0.3,  linewidth=0, edgecolor='none', facecolor=facecolor)
                ax.add_patch(rect)
            rect = patches.Rectangle(
                (p[0], p[1]), p[2], p[3], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        if show_evaluation:
            df_labels = item["labels"]

            df_false_negative = df_labels[(
                df_labels.id.isin(df.gt_id.unique()) == False)]
            for index, row in df_false_negative.iterrows():
                p = row[["bb_left", "bb_top", "bb_width", "bb_height"]].values
                rect = patches.Rectangle(
                    (p[0], p[1]), p[2], p[3], alpha=0.3,  linewidth=1, edgecolor='white', facecolor="orange")
                ax.add_patch(rect)

        if show:
            plt.show()
        return ax

    def plot_segmentations(self,  key, ax=None, show=True):
        item = self.__getitem__(key, ["segmentation"])
        segmentation = item["segmentation"]
        if ax is None:
            ax = plt.gca()
        ax.imshow(segmentation, alpha=0.5)

        if show:
            plt.show()
        return ax

    def plot_masks(self, key, ax=None, show=True, ids=None,):
        item = self.__getitem__(key, ["masks"])
        masks = item["masks"]

        if ax is None:
            ax = plt.gca()
        if ids is not None:
            masks = masks[masks.id.isin(ids)]
        for index, row in masks.iterrows():

            mask = coco_mask.decode(
                {"size": [row["height"], row["width"]], "counts": row["mask"]})
            # segmentation = np.array(Image.fromarray(mask).resize(((960, 576)), Image.NEAREST))

            mask_final = np.zeros((1080, 1920, 4))
            mask_final[:, :, -1] = mask
            mask_final[..., :3] = np.array(
                [*panoptic_loader.id_colors[int(row.id - 2000)]])/255.
            ax.imshow(mask_final, alpha=0.5)

        if show:
            plt.show()
        return ax

    def plot_labels(self,  key, ax=None, show=True, ids=None, id_matching=False,
                    min_visibility=None, max_visibility=None,
                    min_IOU=None, max_IOU=None,
                    free=None,  status=None, color_map=None):

        if id_matching:
            self.match_dets(right="labels")
        item = self.__getitem__(key, fields=["labels"])
        labels = item["labels"]
        if ax is None:
            ax = plt.gca()

        if ids is not None:
            labels = labels[labels.id.isin(ids)]

            assert len(labels) > 0, "IDs `{}` not in sequence".format(
                ", ".join([str(t) for t in ids]))
        if min_visibility is not None:
            assert 0 <= min_visibility <= 1, "`min_visbility has to be in range [0, 1]`"
            labels = labels[labels.visibility >= min_visibility]
        if max_visibility is not None:
            assert 0 <= max_visibility <= 1, "`max_visbility has to be in range [0, 1]`"
            labels = labels[labels.visibility <= max_visibility]

        if min_IOU is not None:

            assert 0 <= min_IOU <= 1, "`min_IOU has to be in range [0, 1]`"
            labels = labels[labels.IOU >= min_IOU]
        if max_IOU is not None:

            assert 0 <= max_IOU <= 1, "`max_IOU has to be in range [0, 1]`"
            labels = labels[labels.IOU <= max_IOU]
        if status is not None:
            labels = labels[labels.status.isin(status)]
        if free is not None:
            labels = labels[labels.free == free]

        for index, row in labels.iterrows():

            p = row[["bb_left", "bb_top", "bb_width", "bb_height"]].values
            if id_matching:
                if row.id_dets > -1:

                    color_id = int(row.id_dets)
                    if color_map is None:
                        color = np.array(
                            [*panoptic_loader.id_colors[color_id], 255])/255.
                    else:
                        color = np.array(
                            [*color_map[color_id], 255])/255.

                else:
                    color = np.ones(4)
            else:
                if color_map is None:
                    color = np.array(
                        [*panoptic_loader.id_colors[int(row.id)], 255])/255.
                else:
                    color = np.array(
                        [*color_map[row.id], 255])/255.

            rect = patches.Rectangle(
                (p[0], p[1]), p[2], p[3], linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # ax.text(x = p[0], y=p[1], s= "{}".format(row.id), fontsize = 30)

        if show:
            plt.show()
        return ax

    def plot_pose_trajectory(self, key, plot_trajectory=True):
        from matplotlib import cm
        evenly_spaced_interval = np.linspace(0, 1, len(self.pose_trajectory))
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]
        if self.pose_trajectory is not None:
            for index, p in enumerate(self.pose_trajectory):
                plt.plot(p[0], p[1], ".", color=colors[index])
        item = self.__getitem__(key)

        pose = item["pose"]
        plt.plot(pose[0, -1], pose[1, -1], "x", markersize=20)
        plt.show()

    def get_pose_trajectory(self):
        pose_list = []
        for data in self:
            pose_list.append(data["pose"][:3, -1])

        self.pose_trajectory = np.stack(pose_list)
        return self.pose_trajectory

    def __iter__(self):
        self._i = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, key, fields=None):
        if fields is None:
            fields = self.frame_cb.keys()

        return {field: callback(self, key) for field, callback in self.frame_cb.items() if field in fields}

    def __next__(self):
        if self._i >= len(self):
            raise StopIteration

        out = self[self._i]
        self._i += 1
        return out
