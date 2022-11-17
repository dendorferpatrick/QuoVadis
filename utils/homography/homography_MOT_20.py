import traceback
import pandas as pd
import os
import copy
import numpy as np
import multiprocessing
import itertools
import open3d as o3d
import itertools
import scipy.spatial.transform as S
from pyntcloud import PyntCloud
import cv2
from tqdm import tqdm 
import sys
from scipy import ndimage, misc

sys.path.append("/usr/wiss/dendorfp/dvl/projects/TrackingMOT/datasets")

from MOT import MOTTracking
import json
from scipy.ndimage.measurements import label


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
    r_plane = RansacPlane(max_dist = 1)
#     r_plane.fit(open3d_point_cloud.points) # or 
    r_plane.least_squares_fit(open3d_point_cloud.points)
    
    return (r_plane.normal, r_plane.point)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Camera:
    def __init__(self, Rt = None, K = None):
        
        assert Rt is not None, "Rt missing"
        assert K is not None, "K is missing"
        self.R = Rt[:3, :3]
        self.t = Rt[:3, 3]
        self.Rt = Rt[:3, :]
        self.K = K        
        self.P = self.K.dot(self.Rt)

def run_homography(sequence_nr, moving = 0):
    try:
        challenge = "MOT20"
        motsynth = MOTTracking(partition = ["{}".format(sequence_nr)], 
                                    challenge = challenge, 
                                    fields=["rgb",
                                    "lidar",
                                    "depth", 
                                    "panoptic", 
                                    "calibration"])
                        
        seq = motsynth.data.sequences[0]
        stop = False
        transformations = {}
        
        for frame in tqdm(range(1, seq.__len__() + 1)):
            try:
                if stop:
                    break
                # labels = copy.deepcopy(seq.labels)
                
                item = seq.__getitem__(frame , ["panoptic", "calibration"])
                calibration = item["calibration"]
                panoptic = item["panoptic"]["mask"]
                mask_shape = panoptic.shape
        

                mask = ((panoptic == 8) | (panoptic == 7) | (panoptic == 6) |  (panoptic == 0))
                
                # mask = ((panoptic == 8) | (panoptic == 7) | (panoptic == 6)).reshape(-1)
                
                structure = np.ones((3, 3), dtype=np.int)
                
                labeled, ncomponents = label(mask*1, structure)
                nr_pixels_com = [] 
                
                for k in range(1,  ncomponents + 1):
                    nr_pixels_com.append(np.sum(labeled == k)) 
                if (np.max(nr_pixels_com) / np.sum( mask ) )> 0.75:
                  
                    mask = (labeled == (np.argmax(nr_pixels_com) + 1))
                mask = ndimage.minimum_filter(mask, size=70)
                mask = mask.reshape(-1)
                pixels = np.array(list(itertools.product(range(mask_shape[0]), range(mask_shape[1]))))
                
                
                mask_p = (np.zeros((mask_shape[0], mask_shape[1])) == 0) 
                # mask_p[:, :int(0.3 *mask_shape[1])] = False
                # mask_p[:700,:] = False
                # mask_p[:, int((1- 0.3) * mask_shape[1]) :] = False
                # mask_p[:int((1- 0.5) * mask_shape[0]), :] = False
            
                mask = np.logical_and(mask, mask_p.reshape(-1))
                
                points, colors, _, img, new_cloud = motsynth.data.sequences[0].transform_lidar_world(frame, transform = False)

            
                
                
                points = points[mask]
                colors = colors[mask]
                pixels = pixels[mask]
                points[:, 1]*=-1
            

                new_cloud.points =  o3d.utility.Vector3dVector(points)
                new_cloud.colors =  o3d.utility.Vector3dVector(colors)


                cloud = PyntCloud.from_instance("open3d", new_cloud)
                is_floor = cloud.add_scalar_field("plane_fit", max_dist = 0.3)

                df = cloud.__dict__["_PyntCloud__points"]
                mask_plane = ( df.is_plane == 1 )
    
                points = points[mask_plane]
                colors = colors[mask_plane]
                pixels = pixels[mask_plane]
                
            
                new_cloud.points =  o3d.utility.Vector3dVector(points)
                new_cloud.colors =  o3d.utility.Vector3dVector(colors)
                        
                normal, origin = get_best_fit_plane(new_cloud, max_dist = 0.3)
                points = np.array(new_cloud.points ) - origin 
                normal*= np.sign(normal[1])
                print(normal, origin)
                # point = points - np.array([0, 0, np.min(points[:, -1])])
                new_cloud.points =  o3d.utility.Vector3dVector(points)

                
                rotation_axis = np.cross( np.array([0,1, 0]), normal)

                rotation_axis/=np.sqrt(np.sum(rotation_axis**2))

                


                rotation_radians = - compute_angle(normal,  np.array([0,1, 0]))

                rotation_vector = rotation_radians * rotation_axis
                rotation = S.Rotation.from_rotvec(rotation_vector)
                rotated_vec = rotation.apply(points)
                
                new_cloud.points =  o3d.utility.Vector3dVector(rotated_vec)
                    
                points = np.array(new_cloud.points )
                # points[:, 2]=points[:, 2] - np.min(points[:, 2])
                # points[:, 0]=points[:, 0] - np.min(points[:, 0])
                new_cloud.points =  o3d.utility.Vector3dVector(points)
                rotated_vec = np.array(new_cloud.points) 
                rotated_vec  = rotated_vec + origin

                points = rotated_vec[:, (0, 2)]
              
                new_pixels = pixels[:, (1,0)]
          
                assert len(points) > 1000, "Not enough points seq {}".format(sequence_nr)
                points[: , 1]-= np.min(points[:, 1])
               
                # points[: , 1]-= np.min(points[:, 1])
                # points[:, 1]=points[:, 1] + np.min(points[:, 1])
                H, status  = cv2.findHomography(new_pixels, points, cv2.RANSAC,2)
                print(H)
                # fx, fy = calibration.get_focal_length()
                # (cx, cy) = calibration.get_principal_point()


                # # labels["u"] = fx * labels.x / labels.y + cx
                # # labels["v"] = fy * labels.z / labels.y + cy
                # labels["u"] = labels.bb_left + labels.bb_width / 2.
                # labels["v"] = labels.bb_top + labels.bb_height 


                # uv = labels[["u", "v"]].values
                # vector_mat = np.concatenate(
                #     (uv, np.ones((len(uv), 1))), axis=-1)
                # trans = np.dot(H, vector_mat.T)


                # trans = trans/trans[-1, :]

                # xy = trans[:2].T

                # labels[["H_x", "H_y"]] = xy
                
                # u = labels["u"].values
                # v = labels.bb_top.values
                
                # mask_v = ((v > 0) & (labels.v < mask_shape[0]))
                # u = u[mask_v] 
                # v = v[mask_v]
                
                # uv = np.stack((u, v),  0).T
                # xy = xy[mask_v]
                
                
                # H_bb, status  = cv2.findHomography( xy, uv, cv2.RANSAC,2.0)
                
                if moving == 0:
                    transformations[-1] =  {
                                        "IPM": H, 
                                        "inv_IPM": np.linalg.inv(H), 
                                        # "IPM_BB": np.linalg.inv(H_bb), 
                                        # "inv_IPM_BB": H_bb
                                        }
                                        
                    stop = True
                else:
                    transformations[frame] =  {
                                        "IPM": H, 
                                        "inv_IPM": np.linalg.inv(H), 
                                        # "IPM_BB": np.linalg.inv(H_bb), 
                                        # "inv_IPM_BB": H_bb
                                        }
                print("/storage/user/dendorfp/{}/homographies_depth/{}-{}.json".format(challenge, challenge, sequence_nr))
                with open("/storage/user/dendorfp/{}/homographies_depth/{}-{}.json".format(challenge, challenge, sequence_nr), 'w') as fp:
                  
                    json.dump(transformations, fp, cls=NumpyEncoder)
            except:
                print(traceback.print_exc())
                pass
            # df = labels
            
            # df.to_csv("/storage/user/dendorfp/MOT16/positions_gt/MOT16-{}.txt".format(sequence_nr), index = False)
            
       
    # error = df.error_H.median()
    # horizon_bottom = df.horizon_bottom.mean()
    # horizon_top = df.horizon_top.mean()
    # row = [sequence_nr, str(horizon_bottom), str(horizon_top), str(error)]
    # print(error)
    # f = open("/storage/user/dendorfp/MOTSynth/statistics/stats.txt", "a")
    # f.write(",".join(row) + "\n")
    # f.close()
    except: 
        print(traceback.print_exc())
        print("Sequence", sequence_nr)
    print("FINISHED")
if __name__ == "__main__":

    # # df_stats = pd.read_csv("/storage/user/dendorfp/MOTSynth/statistics/stats.txt", names = ["sequence", "H_L", "H_T", "H_E"])
    # print(data)
    seq_allowed = ["04", "05", "06", "07", "08"][:1]
    # print(len(seq_allowed), len(seq_df), len(data))
    # # required_sequences = set(seq_allowed).difference(set(df_stats.sequence))
    
    # pool = multiprocessing.Pool(processes=7)
    # pool.map(run_homography, [seq for seq in seq_allowed])
    [run_homography( seq) for seq in seq_allowed]