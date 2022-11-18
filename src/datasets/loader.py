import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import open3d as o3d
from argparse import Namespace
from PIL import Image
import pandas as pd
from typing import List
import pickle5 as pickle
from matplotlib import cm

import subprocess
from tqdm import tqdm

import glob


""" Coordinates are given in image (u [height], v [width] )  and world coordiantes (x, y) """


class SequenceLoader():
    def __init__(self, file, tracker=None):

        if tracker:
            self.columns = ["frame", "id", "bb_left", "bb_top", "bb_width",
                            "bb_height"]
            self.data = pd.read_csv(
                file, names=self.columns, delimiter=",", usecols=(0, 1, 2, 3, 4, 5))

        else:
            self.columns = ["frame", "id", "bb_left", "bb_top", "bb_width",
                            "bb_height", "confidence",  "object_class", "visibility"]
            self.data = pd.read_csv(file, names=self.columns, delimiter=",")
            self.data = self.data[self.data.object_class == 1]

        self.data["bb_center"] = self.data["bb_left"] + \
            self.data["bb_width"]//2
        self.data["bb_bottom"] = self.data["bb_top"] + self.data["bb_height"]
        self.data[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]] = self.data[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]].astype('int32')
    def get_frame(self, frame: int, pedestrian: bool = True):
        frame_data = self.data[(self.data.frame == frame)]
        # if pedestrian:
        #     frame_data = frame_data[frame_data.object_class == 1]
        return frame_data

    def get_positions(self, frame: int, pedestrian: bool = True):
        frame_data = self.get_frame(frame, pedestrian)
        position_data = frame_data[["bb_bottom", "bb_center"]].values
        return position_data, frame_data["id"].values

    def get_id(self, id):
        pass


class HomographyLoader():
    def __init__(self, file):

        with open(file, "rb") as input_file:
            self.homographies = pickle.load(input_file)

    def transform(self, input_data, transformation_type="real2img", homography_matrix = None):
        """ 
        Transforms pixels to world coordinates and vice versa using the homography matrix:
            'img2real': (v, u) => (x, y)
            'real2img': (x, y) => (v, u) 
        """
        assert transformation_type in ["real2img", "img2real", "real2imgGT",
                                       "img2realGT"], "Transformation %s does not exist: 'real2img', 'img2real'"

        vector_mat = np.concatenate(
            (input_data, np.ones((len(input_data), 1))), axis=-1)
        if homography_matrix is not None:
            
            trans = np.dot(homography_matrix, vector_mat.T)
        else:
            trans = np.dot(self.homographies[transformation_type], vector_mat.T)

        # trans[2, :] = -np.maximum(0.001, abs(trans[2, :]))
        trans = np.transpose(trans, (1, 0))

        trans = trans/trans[:, -1:]

        return trans[:, :2]


class PanopticLoader():
    def __init__(self, classes2category_json='./src/datasets/classes2category.json',
                 meta_data_classes_pkl="./src/datasets/metaDataClasses.pkl"):
        
        with open(meta_data_classes_pkl, 'rb') as pkl_file:
            self.meta_data = Namespace(**pickle.load(pkl_file))

        import json
        # Opening JSON file
        with open(classes2category_json) as json_file:
            self.classes2category = json.load(json_file)
        self.category_colors = { "sky": np.array([70, 130, 180]),
                                "pedestrian": np.array([255, 0, 0]),
                                 "other": np.array([255, 215, 0, ]), 
                                 "occluder_moving": np.array([0, 100, 100]),
                                "occluder_static": np.array([0, 0, 230]),                             
                                "building": np.array([0, 255, 0]),  
                                "road": np.array([50, 50, 50]), 
                                "pavement": np.array([100, 100, 100]),                         
                                 "ground": np.array([10, 200, 10]),}
        self.get_color_to_coco()
        self.get_id_color()

    @property
    def categories(self):
        return list(self.category_colors.keys())
    
    def get_color_to_coco(self):
        self.color_to_coco = {}
        for stuff_class, stuff_color in zip(self.meta_data.stuff_classes, self.meta_data.stuff_colors):
            self.color_to_coco[' '.join(str(e) for e in stuff_color)]  = stuff_class
        for thing_class, thing_color in zip(self.meta_data.thing_classes, self.meta_data.thing_colors):
            self.color_to_coco[' '.join(str(e) for e in thing_color)]  = thing_class
    
    def get_id_color(self):
        import itertools 
        palette = list(itertools.product(np.arange(1,256), repeat=3)) 

        step = 1000000

        self.id_colors = []
        for i in range(step): 
            self.id_colors.extend(palette[i::step])
        # self.id_colors_str = [' '.join(str(e) for e in color) for color in self.id_colors]

    def index_category(self, category):
        return self.categories.index(category)

    def get_boundary_between_categories(self, panoptic_file, categories=[]):

        panoptic_img, panoptic_mask = self.get_panoptic_img(
            panoptic_file=panoptic_file, include_categories=categories)
        selected_mask = panoptic_mask
        
        if len(categories) > 0:
            selected_mask = selected_mask - 1000
            for cat in categories:
                selected_mask[panoptic_mask == self.index_category(
                    cat)] = self.index_category(cat)
        
        difference_tensor_forward = torch.zeros_like(selected_mask)
        difference_tensor_backward = torch.zeros_like(selected_mask)
        shifted_tensor_backward = torch.zeros_like(selected_mask)
        shifted_tensor_forward = torch.zeros_like(selected_mask)

        difference_tensor_forward[1:, 1:] = selected_mask[1:,
                                                          1:] - selected_mask[:-1, : -1]
        difference_tensor_backward[:-1, :-
                                   1] = selected_mask[:-1, :-1] - selected_mask[1:, 1:]

        difference_tensor = abs(difference_tensor_backward) + \
            abs(difference_tensor_forward)
        
        difference_tensor[difference_tensor == 0] = 500
        difference_tensor[difference_tensor <= 50] = 1
        difference_tensor[difference_tensor > 50] = 0
        if len(categories) > 0:
            difference_tensor[panoptic_mask !=
                              self.index_category(categories[0])] = 0
        
        difference_pixels = np.array((difference_tensor == 1).nonzero().int())
        return difference_tensor, difference_pixels
    def get_category_map(self, category_mask):
        # seg = torch.load(panoptic_file, map_location='cpu')

        colors = np.unique(panoptic_png.reshape(-1, panoptic_png.shape[2]), axis=0)
        panoptic_img = np.zeros_like(panoptic_png)
        panoptic_mask = np.zeros((category_mask.shape[0], category_mask.shape[1], 4))
        for color in colors: 
            
            color_str = ' '.join(str(e) for e in color)
            if not color_str in self.color_to_coco:
                continue
            coco_class = self.color_to_coco[color_str]
           
            category = self.classes2category[coco_class]
            if len(include_categories) > 0:
                if not category in include_categories:
                    continue
            
            if len(exclude_categories) > 0:
                if category in exclude_categories:
                    continue
            final_color = self.category_colors[category]
            color_indices = np.where(np.all(panoptic_png == color, axis=-1))
            
            panoptic_img[color_indices] = final_color
            panoptic_mask[color_indices] = self.index_category(category)
            # panoptic_id[panoptic_seg == id] = id


        return panoptic_img, panoptic_mask


    def get_panoptic_img_MOTSynth(self, panoptic_png, exclude_categories: List[str] = [], include_categories: List[str] = []):
        # seg = torch.load(panoptic_file, map_location='cpu')

        colors = np.unique(panoptic_png.reshape(-1, panoptic_png.shape[2]), axis=0)
        panoptic_img = np.zeros_like(panoptic_png)
        panoptic_mask = np.zeros((panoptic_png.shape[0], panoptic_png.shape[1]))
        for color in colors: 
            
            color_str = ' '.join(str(e) for e in color)
            if not color_str in self.color_to_coco:
                continue
            coco_class = self.color_to_coco[color_str]
            
            category = self.classes2category[coco_class]
            
            if len(include_categories) > 0:
                if not category in include_categories:
                    continue
            
            if len(exclude_categories) > 0:
                if category in exclude_categories:
                    continue
            final_color = self.category_colors[category]
            color_indices = np.where(np.all(panoptic_png == color, axis=-1))
            
            panoptic_img[color_indices] = final_color
            panoptic_mask[color_indices] = self.index_category(category)
            # panoptic_id[panoptic_seg == id] = id


        return panoptic_img, panoptic_mask
        # , panoptic_id
    def get_panoptic_img(self, panoptic_data, exclude_categories: List[str] = [], include_categories: List[str] = []):
        # seg = torch.load(panoptic_file, map_location='cpu')
        panoptic_seg, segments_info = panoptic_data.values()
        panoptic_img = torch.zeros(
            panoptic_seg.size()[0], panoptic_seg.size()[1], 3).long()
        panoptic_mask = torch.zeros(
            panoptic_seg.size()[0], panoptic_seg.size()[1]).long()
        panoptic_id = torch.zeros(
            panoptic_seg.size()[0], panoptic_seg.size()[1]).long()
        for info in (segments_info):
            id = info["id"]
            category_id = info["category_id"]
            is_thing = info["isthing"]
            if is_thing:
                coco_class = self.meta_data.thing_classes[category_id]
            else:
                coco_class = self.meta_data.stuff_classes[category_id]
            
            category = self.classes2category[coco_class]
            if len(include_categories) > 0:
                if not category in include_categories:
                    continue

            if len(exclude_categories) > 0:
                if category in exclude_categories:
                    continue
            color = self.category_colors[category]
            
            panoptic_img[panoptic_seg == id] = torch.Tensor(color).long()
            panoptic_mask[panoptic_seg == id] = self.index_category(category)
            panoptic_id[panoptic_seg == id] = id


        return panoptic_img, panoptic_mask, panoptic_id

    def get_pixels(self, panoptic_file, category):
        category_index = self.index_category(category)
        panoptic_img, panoptic_mask, panoptic_ids = self.get_panoptic_img(panoptic_file)

        return np.array((panoptic_mask == category_index).nonzero().float()), np.array(panoptic_ids[panoptic_mask == category_index])

    def check_pixels(self, panoptic_file, category, pixels):
        category_index = self.index_category(category)
        panoptic_img, panoptic_mask, _ = self.get_panoptic_img(panoptic_file)
        check_list = []
        for pix in pixels:
            
            check_list.append(panoptic_mask[int(pix[0]), int(pix[1])] == category_index)
        return check_list

    def plot_panoptic_img(self, panoptic_file, img_file: str = False,  exclude_categories: List[str] = [], include_categories: List[str] = []):

        panoptic_img, _, _ = self.get_panoptic_img(
            panoptic_file, exclude_categories=exclude_categories, include_categories=include_categories)

        plt.figure()
        plt.subplot(1, 1, 1)

        if img_file:
            img = plt.imread(img_file)
            plt.imshow(img)
            plt.imshow(panoptic_img, alpha=0.5)
        else:
            plt.imshow(panoptic_img)

        plt.show()


class ImageLoader():
    def get_image(self, image_file):
        img = plt.imread(image_file)
        return np.array(img)


class DepthLoader():

    def get_depth_values(self, depth_file, pixels, resize=(1920, 1080)):
        """ resize (widht, height) """

        depth_data = np.array(self.get_depth_img(depth_file, resize=resize))
        depth_values = np.array(
            [depth_data[int(index[0]), int(index[1])] for index in pixels])

        return depth_values

    def get_depth_img(self, depth_file, resize=()):

        depth_data = np.load(depth_file)["arr_0"]
        depth_img = Image.fromarray(depth_data)
        if len(resize) > 0:

            depth_img = depth_img.resize(
                (resize[0], resize[1]), Image.ANTIALIAS)
        return depth_img

    def plot_depth_img(self, depth_file, img_file: str = False):
        img = plt.imread(img_file)
        depth_img = np.array(self.get_depth_img(depth_file, img.shape[:2]))
        depth_img = Image.fromarray(
            np.uint8(cm.gist_earth(depth_img/np.max(depth_img))*255))
        plt.figure()
        plt.subplot(1, 1, 1)

        plt.imshow(img)
        plt.imshow(depth_img, alpha=0.7)

        plt.show()


class CoordinateTransformer():
    def __init__(self, sequence, tracker=None):
        self.sequence = sequence
        self.tracker = tracker
        print(tracker)
        self.file_loader = FileLoader(sequence=sequence, tracker=tracker)
        self.depth_loader = DepthLoader()
        self.panoptic_loader = PanopticLoader()
        self.homography_loader = HomographyLoader(
            self.file_loader.homorgraphy_file())
        self.image_loader = ImageLoader()

        if tracker:

            self.sequence_loader = SequenceLoader(
                self.file_loader.tracker_file(), tracker=True)
        else:
            self.sequence_loader = SequenceLoader(
                self.file_loader.sequence_file())

    def get_boundary_between_categories(self, frame, categories=[]):
        return self.panoptic_loader.get_boundary_between_categories(panoptic_file=self.file_loader.panoptic_file(frame), categories=categories)

    def plot_panoptic_img(self, frame,    exclude_categories: List[str] = [], include_categories: List[str] = []):
        return self.panoptic_loader.plot_panoptic_img(
            panoptic_file=self.file_loader.panoptic_file(frame), 
            img_file=self.file_loader.img_file(frame), 
            exclude_categories = exclude_categories, 
            include_categories = include_categories)

    def compute_category_depth_coordinates(self, panoptic_file, category):
        pixels, _ = self.panoptic_loader.get_pixels(
            panoptic_file, category)

        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]

        coordinates = self.homography_loader.transform(
            pixels_arranged, "img2real")
        return coordinates, pixels

    def arrange_pixels(self, pixels):

        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]

        return pixels_arranged

    def pixels_2_coordinates(self, pixels):
        """ convert (v: widht, u: height) => (x,y) """
        return self.homography_loader.transform(pixels, "img2real")

    def compute_category_coordinates(self, panoptic_file, category):
        pixels, _ = self.panoptic_loader.get_pixels(
            panoptic_file, category)

        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]

        coordinates = self.homography_loader.transform(
            pixels_arranged, "img2real")
        return coordinates, pixels

    def compute_ground_coordinates(self, panoptic_file):
        """ returns ground_coordinates [x, y], ground_pixels [u, v]  """
        return self.compute_category_coordinates(panoptic_file, "ground")

    def compute_pedestrian_coordinates(self, frame: int, track_id=False):
        pedestrian_pixels, track_ids = self.sequence_loader.get_positions(
            frame)
        pedestrian_coordinates = np.zeros_like(pedestrian_pixels)
        pedestrian_coordinates[:, 0] = pedestrian_pixels[:, 1]
        pedestrian_coordinates[:, 1] = pedestrian_pixels[:, 0]
        pedestrian_coordinates = self.homography_loader.transform(
            pedestrian_coordinates, "img2real")
        return pedestrian_coordinates, pedestrian_pixels, track_ids

    def get_sequence_coordinates(self):
        sequence_df = self.sequence_loader.data

        if not self.tracker:
            sequence_df = sequence_df[sequence_df.object_class == 1]
        pixels = sequence_df[["bb_bottom", "bb_center"]].values
        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]
        coordinates = self.homography_loader.transform(
            pixels_arranged, "img2real")

        sequence_df["x"] = list(coordinates[:, 0])
        sequence_df["y"] = list(coordinates[:, 1])
        self.sequence_df = sequence_df
        return sequence_df

    def check_occlusion(self):
        self.get_sequence_coordinates()

        frames = self.sequence_df["frame"].unique()
        sorted(frames)
        track_ids = self.sequence_df["id"].unique()
        sorted(track_ids)

        occluder_list = []
        for frame in tqdm(frames):
            pixels = self.sequence_df[self.sequence_df["frame"]==frame][["bb_bottom", "bb_center"]].values
            occluder_list.append(self.panoptic_loader.check_pixels(
                panoptic_file=self.file_loader.panoptic_file(frame=frame),
                category="occluder", pixels =  pixels))
        return np.concatenate(occluder_list)

    def create_video(self):
        self.get_sequence_coordinates()

        frames = self.sequence_df["frame"].unique()
        sorted(frames)
        track_ids = self.sequence_df["id"].unique()
        sorted(track_ids)
        evenly_spaced_interval = np.linspace(0, 1, len(track_ids))
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]
        np.random.shuffle(colors)
        track_id_dict = {track: i for i, track in enumerate(track_ids)}
        x_min = self.sequence_df["x"].min()

        x_max = self.sequence_df["x"].max()
        y_min = self.sequence_df["y"].min()
        y_max = self.sequence_df["y"].max()
        for frame in tqdm(frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(45, 15))
            img = plt.imread(self.file_loader.img_file(frame))

            frame_data = self.sequence_df[self.sequence_df.frame == frame]
            for (index, row) in frame_data.iterrows():
                if row["id"] == 53:
                    continue
                track_data = self.sequence_df[self.sequence_df.id == row["id"]]

                # ax1.plot(track_data["x"], track_data["y"],
                #          color=colors[track_id_dict[row["id"]]])
                ax1.plot(row["x"], row["y"], ".", markersize=30,
                         color=colors[track_id_dict[row["id"]]])

            ax2.imshow(img)
            for (index, row) in frame_data.iterrows():
                track_data = self.sequence_df[self.sequence_df.id == row["id"]]
                # ax2.plot(track_data["bb_center"], track_data["bb_bottom"],
                #          color=colors[track_id_dict[row["id"]]])
                ax2.plot(row["bb_center"], row["bb_bottom"], ".",
                         markersize=30,  color=colors[track_id_dict[row["id"]]])

            
            
            ax1.invert_yaxis()

            # plt.show()
            plt.savefig("../videos/%s-%04d.png" % (self.sequence, frame))
            plt.close()
            plt.cla()
            plt.clf()
        subprocess.call([
            'ffmpeg', '-framerate', '20', '-i', "'../videos/{}-%04d.png'".format(
                self.sequence), '-r', '30', '-pix_fmt', 'yuv420p',
            "'../videos/%s.mp4'" % (self.sequence)
        ])

    def get_ground_transformation(self, frame, n = 1000): 
        try:
            depth_coordinates, ground_coordinates, focal_length = self.get_depth_coordinates(frame)
            indices_depth = np.arange(len(depth_coordinates))
            np.random.shuffle(indices_depth)
            indices_depth = indices_depth[:n]
            
            depth_point_cloud, ground_point_cloud = self.get_point_clouds([depth_coordinates[i] for i in indices_depth], [ground_coordinates[i] for i in indices_depth])
            depth_points_array = np.matrix(depth_point_cloud.points)
            ground_points_array = np.matrix(ground_point_cloud.points) 

            transformation = rigid_transform_3D(ground_points_array, depth_points_array, True)
            
            return transformation, focal_length
        except:
            return None, None

    def plot_pedestrian_coordinates(self, frame: int, show_image=True):

        pedestrian_coordinates, pedestrian_pixels, _ = self.compute_pedestrian_coordinates(
            frame)

        if show_image:
            image_file = self.file_loader.img_file(frame)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        else:
            fig, ax1 = plt.subplots()

        evenly_spaced_interval = np.linspace(0, 1, len(pedestrian_coordinates))
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]

        for p, color in zip(pedestrian_coordinates, colors):
            ax1.plot(p[0], p[1], ".", color=color,  markersize=40)
        ax1.invert_yaxis()

        if image_file:
            img = plt.imread(image_file)
            ax2.imshow(img)
            for p, color in zip(pedestrian_pixels, colors):

                ax2.plot(p[1], p[0], ".", color=color, markersize=30)
        plt.show()

    def plot_ground_coordinates(self, frame, plot_image=True):

        panoptic_file = self.file_loader.panoptic_file(frame)
        if plot_image:
            image_file = self.file_loader.img_file(frame)
        ground_coordinates, ground_pixels = self.compute_ground_coordinates(
            panoptic_file)
        indices = np.arange(len(ground_coordinates))
        np.random.shuffle(indices)
        nr_points = 10
        indices = indices[:nr_points]
        evenly_spaced_interval = np.linspace(0, 1, nr_points)
        colors = [cm.rainbow(x) for x in evenly_spaced_interval]

        if image_file:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        else:
            fig, ax1 = plt.subplots()

        for p, color in zip([ground_coordinates[i] for i in indices], colors):
            ax1.plot(p[0], p[1], ".", color=color,  markersize=40)
        ax1.invert_yaxis()

        if image_file:
            img = plt.imread(image_file)
            ax2.imshow(img)
            for p, color in zip([ground_pixels[i] for i in indices], colors):
                ax2.plot(p[1], p[0], ".", color=color,  markersize=40)
        plt.show()

    def get_focal_length(self,  pixels, coordinates, depth,  transform=(0, 0), max_nr_points=100):
        # assumption centered intrinsics
        # ground_pixels (height, widht)
        transformed_pixels = np.zeros_like(pixels)
        transformed_pixels[:, 0] = pixels[:, 0] + transform[0]
        transformed_pixels[:, 1] = pixels[:, 1] + transform[1]

        indices = np.arange(len(pixels))
        np.random.shuffle(indices)
        indices = indices[:max_nr_points]
        focal_length = self.compute_focal_length(
            coordinates[indices], transformed_pixels[indices], depth[indices])

        return focal_length

    def get_category_depth_coordinates(self, frame, category, focal_length=False, return_pixels=False):
        panoptic_file = self.file_loader.panoptic_file(frame)
        depth_file = self.file_loader.depth_file(frame)
        pixels, ids = self.panoptic_loader.get_pixels(panoptic_file, category)
        depth = self.depth_loader.get_depth_values(depth_file, pixels)
        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]

        width = 1920
        height = 1080


        coordinates = self.compute_depth_coordinates(
            pixels_arranged, depth, focal_length,  transform=(-width/2., -height/2.))
        if return_pixels:
            return coordinates, ids, pixels
        else:
            return coordinates, ids
    def get_category_pixels(self, frame, category): 
        panoptic_file = self.file_loader.panoptic_file(frame)
        pixels, ids = self.panoptic_loader.get_pixels(panoptic_file, category)
        return pixels, ids


    def get_depth_coordinates_from_pixels(self, frame, pixels, focal_length=False):
        """ converts (u, v)  pixels into depth coordinates (x, y) """
        depth_file = self.file_loader.depth_file(frame)
        image_file = self.file_loader.img_file(frame)
        img = plt.imread(image_file)
        [height, width, _] = img.shape
        depth_values = self.depth_loader.get_depth_values(
            depth_file, pixels, resize=(width, height))
        pixels_arranged = np.zeros_like(pixels)
        pixels_arranged[:, 0] = pixels[:, 1]
        pixels_arranged[:, 1] = pixels[:, 0]
        depth_coordinates = self.compute_depth_coordinates(
            pixels=pixels_arranged, depth=depth_values, focal_length=focal_length, transform=(-width / 2, -height/2))

        return depth_coordinates

    def get_depth_coordinates(self, frame):
        panoptic_file = self.file_loader.panoptic_file(frame)
        depth_file = self.file_loader.depth_file(frame)
        ground_coordinates, ground_pixels = self.compute_ground_coordinates(
            panoptic_file)

        image_file = self.file_loader.img_file(frame)
        img = plt.imread(image_file)
        [height, width, _] = img.shape
        depth_values = self.depth_loader.get_depth_values(
            depth_file, ground_pixels, resize=(width, height))
        ground_pixels_arranged = np.zeros_like(ground_pixels)
        

        ground_pixels_arranged[:, 0] = ground_pixels[:, 1]
        ground_pixels_arranged[:, 1] = ground_pixels[:, 0]

        focal_length = self.get_focal_length(
            pixels=ground_pixels_arranged, coordinates=ground_coordinates, depth=depth_values, transform=(-width/2, -height/2))

        depth_coordinates = self.compute_depth_coordinates(
            pixels=ground_pixels_arranged, depth=depth_values, focal_length=focal_length, transform=(-width/2, -height/2))

        return depth_coordinates, ground_coordinates, focal_length

    def compute_depth_coordinates(self, pixels, depth,  focal_length, transform=(0, 0)):
        """ return coordinates (x, y, z) """

        transformed_pixels = np.zeros_like(pixels)
        transformed_pixels[:, 0] = pixels[:, 0] + transform[0]
        transformed_pixels[:, 1] = pixels[:, 1] + transform[1]

        coordinates = np.zeros((len(depth), 3))
        coordinates[:, 0] = depth / focal_length * transformed_pixels[:, 0]
        coordinates[:, 1] = depth / focal_length * transformed_pixels[:, 1]
        coordinates[:, 2] = depth

        return coordinates

    def get_point_clouds(self, depth_coordinates, ground_coordinates):
        depth_point_cloud = o3d.geometry.PointCloud()
        depth_point_cloud.points = o3d.utility.Vector3dVector(
            depth_coordinates)

        ground_point_cloud = o3d.geometry.PointCloud()
        ground_points = np.concatenate(
            (ground_coordinates, np.zeros((len(ground_coordinates), 1))), 1)
        
        ground_point_cloud.points = o3d.utility.Vector3dVector(ground_points)

        return depth_point_cloud, ground_point_cloud

    def get_point_cloud(self, depth_coordinates):
        depth_point_cloud = o3d.geometry.PointCloud()
        depth_point_cloud.points = o3d.utility.Vector3dVector(
            depth_coordinates)
        return depth_point_cloud

    def register_point_cloud(self, pcd_source, pcd_target):
        threshold = 10000
        trans_init = np.asarray([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0.0, 0.0, 0,  1]])

        #  do kabsch
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_source, pcd_target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=2000)
        )
        return reg_p2p.transformation

    def compute_focal_length(self, coordinates, pixels, depth_values):
        focal_length_list = []
        for i in np.arange(len(depth_values)):
            for j in np.arange(i+1, len(depth_values)):
                x1 = coordinates[i]
                x2 = coordinates[j]
                u1 = pixels[i]
                u2 = pixels[j]
                z1 = depth_values[i]
                z2 = depth_values[j]

                f = compute_focal_length(x1, x2, u1, u2, z1, z2)
                if not f:
                    continue
                focal_length_list.append(f)
        return np.mean(focal_length_list)

    def warpImage(self, frame, max_dist=50, scale=1000):
        self.get_sequence_coordinates()

        homography = self.homography_loader.homographies["img2real"]
        
        image_file = self.file_loader.img_file(frame)
        image = plt.imread(image_file)

        im_size = image.shape
        x = np.arange(im_size[1])
        y = np.arange(im_size[0])
        mesh_x, mesh_y = np.meshgrid(x, y)
        xyz = np.ones((len(x) * len(y), 3))
        xyz[:, 0] = np.reshape(mesh_x, -1)
        xyz[:, 1] = np.reshape(mesh_y, -1)
        trans = np.dot(homography, xyz.T)

        trans = trans/trans[-1, :]

        xyz = xyz[trans[1, :] < im_size[0], :]
        trans = trans[:, trans[1, :] < im_size[0]]

        xyz = xyz[im_size[0] - trans[1, :] < max_dist, :]
        trans = trans[:, im_size[0] - trans[1, :] < max_dist]
        min_x = np.min(trans[0, :])
        max_x = np.max(trans[0, :])
        min_y = np.min(trans[1, :])
        max_y = np.max(trans[1, :])

        trans[0, :] -= min_x

        trans[1, :] -= min_y

        self.sequence_df["x"] = self.sequence_df["x"] - min_x
        self.sequence_df["y"] = self.sequence_df["y"] - min_y
        #
        # trans = trans[:, trans[1, :] <  70]
        max_y = np.max(trans[1, :])
        trans[1, :] /= max_y
        trans[0, :] /= max_y
        self.sequence_df["x"] = self.sequence_df["x"]/max_y
        self.sequence_df["y"] = self.sequence_df["y"]/max_y

        trans *= scale

        max_x = np.max(trans[0, :])
        max_y = np.max(trans[1, :])

        new_image = np.zeros((int(max_y) + 1, int(max_x) + 1, 3))

        for p, n in zip(xyz, trans[:2].T):

            new_image[int(n[1]), int(n[0])] = image[int(p[1]), int(p[0])]

        M, N, C = new_image.shape
        K = 2

        MK = M // K
        NL = N // K
        new_image = new_image[:MK*K, :NL*K,
                              :].reshape(MK, K, NL, K, C).max(axis=(1, 3))

        # self.data["x"] = self.data["x"] * scale/K
        # self.data["y"] = self.data["y"] * scale/K
        return new_image/255.

    def compute_stablized_coordinates(self, max_nr_frames = False, number_of_occluder_pixels = 1000):
        df = self.get_sequence_coordinates()
        frames = df["frame"].unique() 
        if max_nr_frames: 
            frames = frames[:max_nr_frames]

        depth_coordinates = {}
        inv_transformation = {} 
        focal_lengths_dict = {} 
        for i, frame in tqdm(enumerate(frames)):
            transformation, focal_length = self.get_ground_transformation(frame)
            depth_coordinates[frame] = {} 
            if transformation is None:
                focal_lengths_dict[frame] = None
                
                continue
            focal_lengths_dict[frame] = focal_length

        
            occluder_coordinates, ids, occluder_pixels  = self.get_category_depth_coordinates(frame = frame, 
                                                                        category="occluder", 
                                                                        focal_length = focal_length, 
                                                                        return_pixels = True)    
            
            occluder_index = np.arange(len(occluder_pixels) ) 
            np.random.shuffle(occluder_index)
            occluder_index = occluder_index[:number_of_occluder_pixels]
            
            selected_occluder_pixels = occluder_pixels[occluder_index]
            occluder_coordinates_old_before = occluder_coordinates[occluder_index]        
            occluder_coordinates_cloud = self.get_point_cloud(occluder_coordinates_old_before)
            occluder_coordinates_old = occluder_coordinates_cloud.transform(transformation)

            occluder_coordinates_new = self.get_depth_coordinates_from_pixels(frame, selected_occluder_pixels, focal_length)
            occluder_coordinates_cloud = self.get_point_cloud(occluder_coordinates_new)
            occluder_coordinates_new = occluder_coordinates_cloud.transform(transformation)
            
            occluder_transformation  = rigid_transform_3D(
                np.matrix(occluder_coordinates_old.points), 
                np.matrix(occluder_coordinates_new.points), True)
            occluder_coordinates_old = occluder_coordinates_new
            
        
            
            pedestrian_coordinates, ids, pixels  = self.get_category_depth_coordinates(frame = frame, 
                                                                                category="pedestrian", 
                                                                                focal_length = focal_length, 
                                                                                return_pixels = True)

            pedestrian_coordinates_cloud = self.get_point_cloud(pedestrian_coordinates)
            pedestrian_coordinates_cloud_transformed = pedestrian_coordinates_cloud.transform(transformation)
            
            
            if i != 0:
                pedestrian_coordinates_cloud_transformed = pedestrian_coordinates_cloud_transformed.transform(occluder_transformation)
                complete_transformation = np.matmul(occluder_transformation, transformation)
            else: 
                complete_transformation = transformation
           
            depth_pedestrian_array = np.array(pedestrian_coordinates_cloud_transformed.points)
            inverse_transformation = np.linalg.inv(complete_transformation) 
            inv_transformation[frame] = inverse_transformation
           


            unique_ids = np.unique(ids)
            
            for id in unique_ids:
        

                coordinates = depth_pedestrian_array[ids == id]
                
                coordinates = coordinates[coordinates[:, 1].argsort()[::-1]]
                
                mean_coordinate = np.mean(coordinates[:int(0.05* len(coordinates))], 0) 
                depth_coordinates[frame][id] = mean_coordinate
               
        return depth_coordinates, inv_transformation, focal_lengths_dict 
    def extract_pedestrian_bb(self, save = False, matching = False):
        frames = self.file_loader.nr_panoptic_files()
        print("START BOUNDING BOX EXTRACTIONFROM PANOPTIC SEGMENTATION")
        bbdata_list = [ ]
        for frame in tqdm(frames):
            
            pixels, ids = self.get_category_pixels(frame = frame , category = "pedestrian")
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
            bbdata_list.append(dfuv[["frame", "id" , 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values) 
        bb_data = np.concatenate(bbdata_list)
        bb_df = pd.DataFrame(bb_data, columns = ["frame", "id" , 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
       
        if matching: 
            bb_df["matched_id"] = None
            gt_data =  self.get_sequence_coordinates().values[:, :6]
            matches = id_matching(gt_data, bb_data)
            for frame, match_list in matches.items():
                for id_tracker, id_gt in match_list.items():
                    bb_df.loc[((bb_df.frame == frame) & (bb_df.id == id_tracker)), "matched_id"] = int(id_gt)
        if save: 
            panoptic_bb_file = self.file_loader.panoptic_bb_file()
            output_dir = "/".join(panoptic_bb_file.split("/")[:-1])
            
            os.makedirs(output_dir, exist_ok=True)
           
            bb_df.to_csv(panoptic_bb_file)
        
            

class FileLoader():
    def __init__(self, sequence, data_directory="/usr/wiss/dendorfp/dvl/projects/TrackingMOT/data",
                 homography_directory="/usr/wiss/dendorfp/dvl/projects/TrackingMOT/trackers/GMOTv2",
                 tracker_directory="/usr/wiss/dendorfp/dvl/projects/TrackingMOT/trackers",
                 tracker=None):
        self.sequence = sequence
        self.data_directory = data_directory
        self.homography_directory = homography_directory
        self.tracker_directory = tracker_directory
        self.tracker = tracker

    def img_file(self, frame: int):
        return "{}/{}/img1/{:06d}.jpg".format(self.data_directory, self.sequence, frame)

    def depth_file(self, frame: int):
        return "{}/{}/vlad_depth.pt/{:06d}.npz".format(self.data_directory, self.sequence, frame)

    def nr_panoptic_files(self): 
        
        filePaths = os.listdir("{}/{}/panoptic".format(self.data_directory, self.sequence,))
        
        filePaths = glob.glob("{}/{}/panoptic/*.pkl".format(self.data_directory, self.sequence,))
        files = [int(f.split("/")[-1].split(".")[0] ) for f in filePaths]
        files = sorted(files)
        
        return files

    def panoptic_bb_file(self):
        return "{}/{}/panoptic_bb/{}.csv".format(self.data_directory, self.sequence,  self.sequence)


    def panoptic_file(self, frame: int):
        return "{}/{}/panoptic/{:06d}.pkl".format(self.data_directory, self.sequence, frame)

    def homorgraphy_file(self):
        if self.tracker:
            return "{}/H-{}.pkl".format(self.homography_directory, self.sequence)
        else:
            return "{}/homographies/H-{}.pkl".format(self.data_directory, self.sequence)

    def sequence_file(self):
        return "{}/{}/gt/gt.txt".format(self.data_directory, self.sequence)


    def tracker_file(self):
        return "{}/{}/{}.txt".format(self.tracker_directory, self.tracker, self.sequence)


if __name__ == "__main__":
    sequence = "MOT16-04"
    coordinate_transformer = CoordinateTransformer(sequence)
    coordinate_transformer.extract_pedestrian_bb(save =  True, matching = True)
