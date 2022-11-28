import itertools
import json

import cv2
import numpy as np
import open3d as o3d
import scipy.spatial.transform as S

from quovadis.datasets.MOT import MOTData
from pyntcloud import PyntCloud
from pyntcloud.ransac.models import RansacPlane
from scipy import ndimage
from scipy.ndimage.measurements import label
from tqdm import tqdm


def compute_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Computes angle between two vectors
    Args:
        u: first vector
        v: second vector
    Returns:
        angle between the vectors in radians
    """
    return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def get_best_fit_plane(open3d_point_cloud, max_dist=1):
    r_plane = RansacPlane(max_dist=max_dist)
    r_plane.least_squares_fit(open3d_point_cloud.points)

    return (r_plane.normal, r_plane.point)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Camera:
    def __init__(self, Rt=None, K=None):

        assert Rt is not None, "Rt missing"
        assert K is not None, "K is missing"
        self.R = Rt[:3, :3]
        self.t = Rt[:3, 3]
        self.Rt = Rt[:3, :]
        self.K = K
        self.P = self.K.dot(self.Rt)


def run_homography(dataset: str, sequence: str, moving=0):
    mot = MOTData(sequences=[sequence],
                  dataset=dataset,
                  fields=["rgb",
                          "depth",
                          "panoptic",
                          "calibration"])
    seq = mot.data.sequences[0]
    stop = False
    transformations = {}

    for frame in tqdm(range(1, seq.__len__() + 1)):

        if stop:
            break
        item = seq.__getitem__(frame, ["panoptic"])
        panoptic = item["panoptic"]["mask"]
        mask_shape = panoptic.shape

        # filter all ground pixels
        mask = ((panoptic == 8) | (panoptic == 7) |
                (panoptic == 6) | (panoptic == 0))

        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(mask*1, structure)
        nr_pixels_com = []

        for k in range(1,  ncomponents + 1):
            nr_pixels_com.append(np.sum(labeled == k))

        if (np.max(nr_pixels_com) / np.sum(mask)) > 0.75:

            mask = (labeled == (np.argmax(nr_pixels_com) + 1))
        mask = ndimage.minimum_filter(mask, size=70)

        pixels = np.array(list(itertools.product(
            range(mask_shape[0]), range(mask_shape[1]))))
        mask_p = (np.zeros((mask_shape[0], mask_shape[1])) == 0)
        mask = np.logical_and(mask.reshape(-1), mask_p.reshape(-1))

        # creating point cloud from depth map
        points, colors, _, img, new_cloud = mot.data.sequences[0].transform_depth_world(
            frame, transform=False)

        points = points[mask]
        colors = colors[mask]
        pixels = pixels[mask]
        points[:, 1] *= -1

        new_cloud.points = o3d.utility.Vector3dVector(points)
        new_cloud.colors = o3d.utility.Vector3dVector(colors)

        cloud = PyntCloud.from_instance("open3d", new_cloud)

        df = cloud.__dict__["_PyntCloud__points"]
        mask_plane = (df.is_plane == 1)

        points = points[mask_plane]
        colors = colors[mask_plane]
        pixels = pixels[mask_plane]

        new_cloud.points = o3d.utility.Vector3dVector(points)
        new_cloud.colors = o3d.utility.Vector3dVector(colors)

        normal, origin = get_best_fit_plane(new_cloud, max_dist=0.3)
        points = np.array(new_cloud.points) - origin
        normal *= np.sign(normal[1])

        new_cloud.points = o3d.utility.Vector3dVector(points)

        rotation_axis = np.cross(np.array([0, 1, 0]), normal)

        rotation_axis /= np.sqrt(np.sum(rotation_axis**2))

        rotation_radians = - \
            compute_angle(normal,  np.array([0, 1, 0]))

        rotation_vector = rotation_radians * rotation_axis
        rotation = S.Rotation.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(points)

        new_cloud.points = o3d.utility.Vector3dVector(rotated_vec)

        points = np.array(new_cloud.points)
        new_cloud.points = o3d.utility.Vector3dVector(points)
        rotated_vec = np.array(new_cloud.points)
        rotated_vec = rotated_vec + origin

        points = rotated_vec[:, (0, 2)]

        new_pixels = pixels[:, (1, 0)]

        assert len(points) > 1000, "Not enough points seq {}".format(
            sequence)
        points[:, 1] -= np.min(points[:, 1])

        H, status = cv2.findHomography(
            new_pixels, points, cv2.RANSAC, 2)

        if moving == 0:
            transformations[-1] = {
                "IPM": H,
                "inv_IPM": np.linalg.inv(H),

            }

            stop = True
        else:
            transformations[frame] = {
                "IPM": H,
                "inv_IPM": np.linalg.inv(H),

            }
        o3d.visualization.draw(new_cloud)
        with open(f'./data/{dataset}/sequences/{dataset}/{sequence}/homography/homography.json', 'w') as fp:
            json.dump(transformations, fp, cls=NumpyEncoder)
