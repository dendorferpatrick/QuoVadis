import pickle
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import torch
from tqdm import tqdm


class PanopticLoader():
    def __init__(self, classes2category_json='./src/quovadis/datasets/classes2category.json',
                 meta_data_classes_pkl="./src/quovadis/datasets/metaDataClasses.pkl"):

        with open(meta_data_classes_pkl, 'rb') as pkl_file:
            self.meta_data = Namespace(**pickle.load(pkl_file))

        import json

        # Opening JSON file
        with open(classes2category_json) as json_file:
            self.classes2category = json.load(json_file)
        self.category_colors = {"sky": np.array([70, 130, 180]),
                                "pedestrian": np.array([255, 0, 0]),
                                "other": np.array([255, 215, 0, ]),
                                "occluder_moving": np.array([0, 100, 100]),
                                "occluder_static": np.array([0, 0, 230]),
                                "building": np.array([0, 255, 0]),
                                "road": np.array([50, 50, 50]),
                                "pavement": np.array([100, 100, 100]),
                                "ground": np.array([10, 200, 10]), }
        self.get_color_to_coco()
        self.get_id_color()

    @property
    def categories(self):
        return list(self.category_colors.keys())

    def get_color_to_coco(self):
        self.color_to_coco = {}
        for stuff_class, stuff_color in zip(self.meta_data.stuff_classes, self.meta_data.stuff_colors):
            self.color_to_coco[' '.join(str(e)
                                        for e in stuff_color)] = stuff_class
        for thing_class, thing_color in zip(self.meta_data.thing_classes, self.meta_data.thing_colors):
            self.color_to_coco[' '.join(str(e)
                                        for e in thing_color)] = thing_class

    def get_id_color(self):
        import itertools
        palette = list(itertools.product(np.arange(1, 256), repeat=3))

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

        colors = np.unique(
            panoptic_png.reshape(-1, panoptic_png.shape[2]), axis=0)
        panoptic_img = np.zeros_like(panoptic_png)
        panoptic_mask = np.zeros(
            (category_mask.shape[0], category_mask.shape[1], 4))
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
        panoptic_img, panoptic_mask, panoptic_ids = self.get_panoptic_img(
            panoptic_file)

        return np.array((panoptic_mask == category_index).nonzero().float()), np.array(panoptic_ids[panoptic_mask == category_index])

    def check_pixels(self, panoptic_file, category, pixels):
        category_index = self.index_category(category)
        panoptic_img, panoptic_mask, _ = self.get_panoptic_img(panoptic_file)
        check_list = []
        for pix in pixels:

            check_list.append(
                panoptic_mask[int(pix[0]), int(pix[1])] == category_index)
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
