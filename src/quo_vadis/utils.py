import argparse
import numpy as np


import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import pandas as pd

from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import yaml

from datasets.utils import pix2real


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


def calculate_similarities(bboxes1, bboxes2, img_shape=None, do_ioa=None):
    similarity_scores = _calculate_box_ious(
        bboxes1, bboxes2, box_format="xywh", img_shape=img_shape, do_ioa=do_ioa)
    return similarity_scores


def ious(bboxes1, bboxes2,  img_shape=None):
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
    intersection[area1 <= 0 + np.finfo('float').eps, ] = 0
    intersection[area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1
    ious = intersection / union
    return ious


def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False, img_shape=None):
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

        ioas *= (-1 + 2 * (bboxes1[:,  np.newaxis, 3]
                 <= bboxes2[np.newaxis, :, 3]))
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


def id_matching(data_1, data_2, img_shape=None):
    """ input data [frame, id, x, y,  w, t] """
    assert data_1.shape[-1] == 6, "data_1 wrong shape"
    assert data_2.shape[-1] == 6, "data_2 wrong shape"
    THRESHOLD = 0.5
    data_1 = data_1.astype(int)
    data_2 = data_2.astype(int)

    matches_list = []
    continguous_dict_1 = {}
    continguous_dict_2 = {}
    for new_id, id in enumerate(np.unique(data_1[:, 1])):
        data_1[data_1[:, 1] == id, 1] = new_id
        continguous_dict_1[new_id] = id
    for new_id, id in enumerate(np.unique(data_2[:, 1])):
        data_2[data_2[:, 1] == id, 1] = new_id
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
            bboxes1=bboxes1[:, 2:], bboxes2=bboxes2[:, 2:], img_shape=img_shape)

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


def id_matching_ioa(data_1, data_2, img_shape=None):
    """ input data [frame, id, x, y,  w, t] """
    assert data_1.shape[-1] == 6, "data_1 wrong shape"
    assert data_2.shape[-1] == 6, "data_2 wrong shape"
    THRESHOLD = 0.5
    data_1 = data_1.astype(int)
    data_2 = data_2.astype(int)

    matches_list = []
    continguous_dict_1 = {}
    continguous_dict_2 = {}
    for new_id, id in enumerate(np.unique(data_1[:, 1])):
        data_1[data_1[:, 1] == id, 1] = new_id
        continguous_dict_1[new_id] = id
    for new_id, id in enumerate(np.unique(data_2[:, 1])):
        data_2[data_2[:, 1] == id, 1] = new_id
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
            bboxes1=bboxes1[:, 2:], bboxes2=bboxes2[:, 2:], img_shape=img_shape, do_ioa=True))

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


def overlap(data,  img_shape=None):
    """ input data [frame, id, x, y,  w, t] """
    assert data.shape[-1] == 6, "data_1 wrong shape"

    data = data.astype(int)

    continguous_dict = {}

    for new_id, id in enumerate(np.unique(data[:, 1])):
        data[data[:, 1] == id, 1] = new_id
        continguous_dict[new_id] = id

    frames = np.unique(data[:, 0])
    matches = []

    for frame in frames:

        bboxes = data[data[:, 0] == frame]

        if len(bboxes) == 0:
            continue

        iou_scores = calculate_similarities(
            bboxes1=bboxes[:, 2:], bboxes2=bboxes[:, 2:], img_shape=img_shape)

        iou_scores -= np.eye(len(bboxes))
        row, col = np.where((iou_scores > 0))

        ioa_scores = calculate_similarities(
            bboxes1=bboxes[:, 2:], bboxes2=bboxes[:, 2:], img_shape=img_shape, do_ioa=True)

        ioa_scores -= np.eye(len(bboxes))

        # for r, c in zip(row, col):
        ids = np.array([continguous_dict[id] for id in bboxes[:, 1]])
        final_iou_scores = iou_scores[row, col]
        final_ioa_scores = ioa_scores[row, col]
        matches.append(np.vstack(((np.ones(len(row)) * frame).astype(int),
                       ids[row], ids[col], final_iou_scores, final_ioa_scores)).T)
    df = pd.DataFrame(np.concatenate(matches), columns=[
                      "frame", "id1", "id2", "IOU", "IOA"])
    df[["frame", "id1", "id2"]] = df[["frame", "id1", "id2"]].astype("int")
    return df


def overlap_frame(data_1, data_2,  img_shape=None):
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
        bboxes1=bboxes1, bboxes2=bboxes2, img_shape=img_shape, do_ioa=True)

    # ioa_scores-=np.eye(len(bboxes))

    # for r, c in zip(row, col):

    # final_iou_scores = iou_scores[row, col]

    final_ioa_scores_max = np.max((ioa_scores), 1)
    final_ioa_scores_min = np.min((ioa_scores), 1)

    return (final_ioa_scores_max > 0) * final_ioa_scores_max + (final_ioa_scores_max <= 0) * final_ioa_scores_min


if __name__ == "__main__":
    x = np.array([[781.3,  444.7,  72.3, 207.5]])
    y = np.array([[808.79619393, 446.74116917, 24.3,      80.]])
    score_mat = overlap_frame(y, x)

    print(score_mat)


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


def load_cfg_file(args):
    with open(args.config_file, "r") as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg_dict


def get_cfg(args):
    cfg_dict = load_cfg_file(args)

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

    cfg = argparse.Namespace(**cfg_dict)
    return cfg


def parse_sequence_arg(sequence_str):
    challenge, sequence_nr_str = sequence_str.split("-")
    sequence_nr = int(sequence_nr_str)
    return challenge, sequence_nr


def compute_dist(X, Y=None, distance_metric='cosine'):
    if Y is None:
        Y = X
    dist = pairwise_distances(
        X,
        Y=Y,
        metric=distance_metric)

    return dist


def L2_threshold(age, min_threshold=1., max_threshold=3., alpha=0.05):
    return np.minimum(min_threshold + alpha * age, max_threshold)


def compute_IOU_scores(x, y, threshold=0.5, img_shape=None):

    x = x.astype(int)
    y = y.astype(int)

    iou_scores = calculate_similarities(
        bboxes1=x, bboxes2=y, img_shape=img_shape)
    score_mat = iou_scores * 1.
    score_mat[score_mat < threshold - np.finfo('float').eps] = 0
    return score_mat, iou_scores


def compute_L2_scores(x, y, min_threshold=1, max_threshold=2, age=None):
    if age is None:
        threshold = max_threshold * np.ones(len(x))
    else:
        threshold = L2_threshold(age, min_threshold,  max_threshold)

    # threshold = threshold * 0 + 0.05
    dist_mat = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    score_mat = np.sqrt(np.sum(dist_mat**2, -1))
    score_mat = np.maximum(
        threshold[:, np.newaxis] - score_mat, 0)/threshold[:, np.newaxis]
    return score_mat


def matching(score_mat):

    # Hungarian algorithm to find best matches
    match_rows, match_cols = linear_sum_assignment(-score_mat)
    actually_matched_mask = score_mat[match_rows,
                                      match_cols] > 0 + np.finfo('float').eps
    match_rows = match_rows[actually_matched_mask]
    match_cols = match_cols[actually_matched_mask]
    return (match_rows, match_cols)



def prepare_sequence(mot, sequence):
    tracker_df = mot.data.sequences[0].tracker

    item = sequence.__getitem__(1, fields=["homography", "map"])

    visibility = item["map"]["visibility_img"]
    height, width = visibility.shape
    tracker_df["u"] = tracker_df["bb_left"] + 1/2.*tracker_df["bb_width"]
    tracker_df["v"] = tracker_df["bb_top"] + tracker_df["bb_height"]
    pixel_positions_raw = tracker_df[["u", "v", "frame"]].values
    # occluded = visibility[np.clip(pixel_positions_raw[:, 1].astype(
    #     int), 0, height - 1), np.clip(pixel_positions_raw[:, 0].astype(int),  0, width - 1)] == 0
    # tracker_df["occluded"] = occluded * 1

    #     ay.plot(pixel_positions[:, 1], pos[:,1] , ".", label ="normal")


    pixel_positions = np.concatenate(
        (pixel_positions_raw[:, :2], np.ones((len(pixel_positions_raw), 1))), 1)

    if sequence.has_egomotion:
        frames = pixel_positions_raw[:, -1].astype(int)
        new_pos = np.zeros_like(pixel_positions)
        for frame in tqdm(np.unique(frames)):
            item = sequence.__getitem__(frame, ["homography", "egomotion"])

            H = np.array(item["homography"]["IPM"])

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
        H = np.array(item["homography"]["IPM"])
        y0 = get_y0(H, width)
        pos = H.dot(pixel_positions.T).T
        pos = pos/pos[:, -1:]

        if y0 is not None:
            new_pos = pix2real(H, pos*1., pixel_positions*1., y0, img_width=width)
        else:
            new_pos = pos*1.

    tracker_df[["x", "z", "y"]] = new_pos
    return tracker_df,y0
