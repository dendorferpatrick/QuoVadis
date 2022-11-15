import copy
import os
import sys
from collections import ChainMap, namedtuple
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from utils import PanopticLoader  # noqa: E2

from . import funcs
from .dataset import DataSet, Sequence

panoptic_loader = PanopticLoader()


class MOTTracking(DataSet):
    """Parsing MOT tracking data."""

    def __init__(self, partition=["02"], challenge="MOT16",    prefix="/storage/user/dendorfp", fields=None):
        super().__init__(prefix)
        self.partition = partition
        self.dataset = challenge
        self.data = self._Partition(
            [challenge + "-" + p for p in partition], os.path.join(self.prefix, challenge), dataset = self.dataset,  fields=fields)
        
    class _Partition:
        Field = namedtuple("Field", ["id", "folder", "setup_cb", "frame_cb"])
        test_sequneces = ["01", "03", "06", "07", "08", "12", "14"]
        _all_fields = {
            "depth": Field("depth", "depth_img", funcs._depth_setup_cb_MOT, funcs._depth_frame_cb_MOT),
            "calibration": Field("calibration", "calib", funcs._calibration_setup_cb_mot, funcs._calibration_frame_cb),
            "labels": Field("labels", "labels", funcs._labels_setup_cb_mot, funcs._labels_frame_cb),
            "lidar": Field("lidar", "depth_img", funcs._depth_setup_cb_MOT, funcs._depth_frame_cb_MOT),
            "pose": Field("pose", "floor_alignment_new", funcs._pose_setup_cb_mot, funcs._pose_frame_cb),
            "rgb": Field("rgb", "img1", funcs._rgb_setup_cb_MOT, funcs._rgb_frame_cb_MOT),
            "dets": Field("dets", "dets", funcs._dets_setup_cb, funcs._dets_frame_cb),
            "panoptic": Field("panoptic", "panoptic", funcs._panoptic_setup_cb, funcs._panoptic_frame_cb),
            "segmentation": Field("segmentation", "segmentation", funcs._segmentation_setup_cb, funcs._segmentation_frame_cb),
            "map": Field("map", "mapping", funcs._map_setup_cb, funcs._map_frame_cb),
            "map_img": Field("map_img", "mapping_img", funcs._map_img_setup_cb, funcs._map_img_frame_cb),
            "positions": Field("positions", "positions_gt", funcs._positions_setup_cb, funcs._positions_frame_cb), 
            "tracker": Field("tracker", "tracker", funcs._tracker_setup_cb_mot, funcs._tracker_frame_cb), 
            "homography_depth": Field("homography_depth", "homographies_depth",  funcs._homography_setup_cb, funcs._homography_depth_frame_cb),
            "homography_bb": Field("homography_bb", "homographies_bb",  funcs._homography_setup_cb, funcs._homography_bb_frame_cb),
            "positions_h_depth_bb": Field("positions_h_depth_bb", "positions_h_depth_bb",  funcs._positions_h_setup_cb, funcs._positions_h_depth_bb_frame_cb),
            "egomotion" : Field("egomotion", "egomotion", funcs._egomotion_cb, funcs._egomotion_frame_cb)
            
        }

        def __init__(self, stage, prefix, dataset, fields=None):
            assert fields is not None, "`fields` cannot be `None`"
            self.stage = stage
            self.prefix = prefix
            self.fields = {}

            
            # check if all fields are available
            for field in fields:
                
                if not os.path.isdir(
                    os.path.join(self.prefix, self._all_fields[field].folder)
                ):

                    raise RuntimeError(
                        f"No data for field: {field}; Missing {os.path.join(self.prefix, self._all_fields[field].folder)}")

                self.fields[field] = self._all_fields[field]

            # if all are present, register callback
            a_key = next(iter(self.fields))

            # TODO this will for fields which use a single file to store info about the
            # the whole sequence

            seq_names = sorted(
                [
                    os.path.splitext(d.name)[0]
                    for d in os.scandir(
                        os.path.join(self.prefix,  self.fields[a_key].folder)
                    )
                ]
            )

            
            self.sequences = [
                Sequence(self.prefix, name, self.fields, dataset = dataset)
                for name in seq_names if name in stage
            ]

        @property
        def sequence_names(self):
            sequence_names = [seq.name for seq in self.sequences]
            return sequence_names

        def __getitem__(self, key):

            print(key)
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
    
    mot = MOTTracking(partition=["01"], challenge = "MOT16" , fields=[
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
                frame, transform = False)
   

    points = points[mask]
    colors = colors[mask]
    points[:, 1]*=-1
    new_cloud.points =  o3d.utility.Vector3dVector(points)
    new_cloud.colors =  o3d.utility.Vector3dVector(colors)
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
    mask_plane = ( df.is_plane == 1 )
    points = points[mask_plane]
    colors = colors[mask_plane]
    new_cloud.points =  o3d.utility.Vector3dVector(points)
    new_cloud.colors =  o3d.utility.Vector3dVector(colors)


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



    def get_best_fit_plane(open3d_point_cloud, max_dist = 1):
        r_plane = RansacPlane(max_dist = max_dist)

        r_plane.fit(open3d_point_cloud.points) # or 
        r_plane.least_squares_fit(open3d_point_cloud.points)
        print(r_plane.__dict__)
        return (r_plane.normal, r_plane.point)

    normal, origin = get_best_fit_plane(new_cloud, 2)

    
    points = np.array(new_cloud.points ) - origin 
    
    
    # point = points - np.array([0, 0, np.min(points[:, -1])])
    new_cloud.points =  o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw(new_cloud)
    c_cloud = copy.deepcopy(new_cloud)
    rotation_axis = np.cross( np.array([0,1, 0]), normal)

    rotation_axis/=np.sqrt(np.sum(rotation_axis**2))

    vec = points

    
    rotation_radians = - compute_angle(normal,  np.array([0,1, 0]))
    
    # print(rotation_radians)

    rotation_vector = rotation_radians * rotation_axis
    rotation = S.Rotation.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)
    
    

    
    new_cloud.points =  o3d.utility.Vector3dVector(rotated_vec)
        
    points = np.array(new_cloud.points )
    points[:, 2]=points[:, 2] - np.min(points[:, 2])
    # points[:, 0]=points[:, 0] - np.min(points[:, 0])
    new_cloud.points =  o3d.utility.Vector3dVector(points)
    o3d.visualization.draw(new_cloud)
