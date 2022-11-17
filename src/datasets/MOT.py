import copy
import os
import sys
from collections import ChainMap, namedtuple
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


from . import utils
from .dataset import Sequence, DataSet

# panoptic_loader = PanopticLoader()


class MOTData(DataSet):
    """Parsing MOT tracking data."""

    def __init__(self, sequences=["MOT17-02"], dataset="MOT17",    prefix="./data", fields=None):
        super().__init__(prefix)
        self.sequences = sequences
        self.dataset = dataset
        
        self.data = self._Partition(sequences, os.path.join(
            self.prefix, dataset), dataset=self.dataset,  fields=fields)

    class _Partition:
        Field = namedtuple(
            "Field", ["id", "folder", "location", "setup_cb", "frame_cb"])

        _all_fields = {
            "depth": Field("depth", "depth_img", "sequences", utils._depth_setup_cb_MOT, utils._depth_frame_cb_MOT),
            "calibration": Field("calibration", "calib", "sequences", utils._calibration_setup_cb_mot, utils._calibration_frame_cb),
            "labels": Field("labels", "labels", "sequences", utils._labels_setup_cb_mot, utils._labels_frame_cb),
            "lidar": Field("lidar", "depth_img", "sequences", utils._depth_setup_cb_MOT, utils._depth_frame_cb_MOT),
            "pose": Field("pose", "floor_alignment_new", "sequences", utils._pose_setup_cb_mot, utils._pose_frame_cb),
            "rgb": Field("rgb", "img1", "sequences", utils._rgb_setup_cb_MOT, utils._rgb_frame_cb_MOT),
            "dets": Field("dets", "det", "sequences", utils._dets_setup_cb, utils._dets_frame_cb),
            "panoptic": Field("panoptic", "panoptic", "sequences", utils._panoptic_setup_cb, utils._panoptic_frame_cb),
            "segmentation": Field("segmentation", "segmentation", "sequences", utils._segmentation_setup_cb, utils._segmentation_frame_cb),
            "map": Field("map", "map", "sequences", utils._map_setup_cb, utils._map_frame_cb),
            "positions": Field("positions", "positions_gt", "sequences", utils._positions_setup_cb, utils._positions_frame_cb),
            "tracker": Field("tracker", "data", "tracker", utils._tracker_setup_cb_mot, utils._tracker_frame_cb),
            "homography": Field("homography", "homography",  "sequences", utils._homography_setup_cb, utils._homography_frame_cb),
            "egomotion": Field("egomotion", "egomotion", "sequences", utils._egomotion_cb, utils._egomotion_frame_cb)

        }

        def __init__(self, sequences_list, prefix, dataset, fields=None):
            assert fields is not None, "`fields` cannot be `None`"
            self.sequences_list = sequences_list
            self.prefix = prefix
            self.fields = {}

            # check if all fields are available
            
            for field in fields: 
                for seq in self.sequences_list:
                    if (self._all_fields[field].id != "tracker"):
                        if not os.path.isdir(
                                os.path.join(self.prefix, self._all_fields[field].location, seq,
                                            self._all_fields[field].folder)
                                ):

                            raise RuntimeError(
                                f"No data for field: {field}; Missing {os.path.join(self.prefix, self._all_fields[field].location,  seq, self._all_fields[field].folder)}")
                    
                self.fields[field] = self._all_fields[field]

            # if all are present, register callback
            a_key = next(iter(self.fields))

            self.sequences = [
                Sequence(self.prefix, name, self.fields, dataset=dataset)
                for name in self.sequences_list
            ]

        @property
        def sequence_names(self):
            sequence_names = [seq.name for seq in self.sequences]
            return sequence_names
        
       

        def __getitem__(self, key):
            return self.sequences[key]

        def __iter__(self):
            return iter(self.sequences)

        def __len__(self):
            return len(self.sequences)


if __name__ == "__main__":

    # df = pd.read_csv(
    #     "/storage/user/dendorfp/MOT16/trajectories/MOT16-02_trajectories.csv")

    # load trajectories
    import scipy.spatial.transform as S
    o3d.visualization.webrtc_server.enable_webrtc()

    mot = MOTTracking(partition=["01"], challenge="MOT16", fields=[
        "rgb",
        "lidar",
        "panoptic",
        "depth",
        "calibration"])
    frame = 59
    item = mot.data.sequences[0].__getitem__(
        frame, ["lidar", "pose", "calibration", "rgb", "lidar", "panoptic"])

    lidar = item["lidar"]
    panoptic = item["panoptic"]["mask"]

    mask_shape = panoptic.shape

    mask = ((panoptic == 8) | (panoptic == 7) | (panoptic == 6)).reshape(-1)

    mask_p = (np.zeros((mask_shape[0], mask_shape[1])) == 0)
    # mask_p[:, :int(0.3 *mask_shape[1])] = False
    # mask_p[:700,:] = False
    # mask_p[:, int((1- 0.3) * mask_shape[1]) :] = False
    mask = np.logical_and(mask, mask_p.reshape(-1))
    points, colors, _, img, new_cloud = mot.data.sequences[0].transform_lidar_world(
        frame, transform=False)

    points = points[mask]
    colors = colors[mask]
    points[:, 1] *= -1
    new_cloud.points = o3d.utility.Vector3dVector(points)
    new_cloud.colors = o3d.utility.Vector3dVector(colors)
    # vis = o3d.visualization.Visualizer()

    # vis.create_window(visible=False) #works for me with False, on some systems needs to be true
    # vis.add_geometry(new_cloud)
    # vis.update_geometry(new_cloud)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("notebooks/3d.png")
    # vis.destroy_window()
    o3d.visualization.draw(new_cloud)

    from pyntcloud import PyntCloud

    cloud = PyntCloud.from_instance("open3d", new_cloud)
    is_floor = cloud.add_scalar_field("plane_fit", max_dist=1)
    print(is_floor)
    print(cloud.__dict__["_PyntCloud__points"])
    df = cloud.__dict__["_PyntCloud__points"]
    mask_plane = (df.is_plane == 1)
    points = points[mask_plane]
    colors = colors[mask_plane]
    new_cloud.points = o3d.utility.Vector3dVector(points)
    new_cloud.colors = o3d.utility.Vector3dVector(colors)

    from pyntcloud.ransac.models import RansacPlane

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

        r_plane.fit(open3d_point_cloud.points)  # or
        r_plane.least_squares_fit(open3d_point_cloud.points)
        print(r_plane.__dict__)
        return (r_plane.normal, r_plane.point)

    normal, origin = get_best_fit_plane(new_cloud, 2)

    points = np.array(new_cloud.points) - origin

    # point = points - np.array([0, 0, np.min(points[:, -1])])
    new_cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw(new_cloud)
    c_cloud = copy.deepcopy(new_cloud)
    rotation_axis = np.cross(np.array([0, 1, 0]), normal)

    rotation_axis /= np.sqrt(np.sum(rotation_axis**2))

    vec = points

    rotation_radians = - compute_angle(normal,  np.array([0, 1, 0]))

    # print(rotation_radians)

    rotation_vector = rotation_radians * rotation_axis
    rotation = S.Rotation.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)

    new_cloud.points = o3d.utility.Vector3dVector(rotated_vec)

    points = np.array(new_cloud.points)
    points[:, 2] = points[:, 2] - np.min(points[:, 2])
    # points[:, 0]=points[:, 0] - np.min(points[:, 0])
    new_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw(new_cloud)

if __name__ == "__main__":
    DataSet(sequences=["MOT17-02"], challenge="MOT17",
            prefix="./data", fields=["rgb"])
