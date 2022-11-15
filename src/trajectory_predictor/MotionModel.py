import copy
import itertools
import logging
import os
import shutil
import sys
import traceback
from argparse import Namespace
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from setuptools import SetuptoolsDeprecationWarning
from torch import dsmm
from tqdm import tqdm

from helper import calculate_similarities, overlap_frame  # noqa: E2

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

def get_y0(H, img_width):

    x_array = np.arange(0, img_width)
    horizon = -(H[2, 0] * x_array + H[2, 2] )/ H[2, 1]
    y0_list = []
    for h, x in zip(horizon, x_array):

        y = np.arange( np.ceil(h)+1, 1080)

        xx = np.ones(len(y)) * x
        p = np.stack((xx, y, np.ones(len(y))))
        pp = H.dot(p).T
        pp = pp[:, :2]/pp[:, -1:]
        dd = pp[1:, 1] - pp[:-1, 1]

        dk = dd[1:]/ dd[:-1]

        pix_y = y[1:]
        lower_threshold = pix_y[abs(dd) > .2]
        
        if len(lower_threshold) == 0 :
            y0_list.append(h+ 40)
        else: y0_list.append(lower_threshold[-1])
    return np.array(y0_list)

def compute_dist(X, Y=None, distance_metric='cosine'):
    if Y is None:
        Y = X
    dist = sklearn.metrics.pairwise_distances(
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
    score_mat = np.maximum(threshold[:, np.newaxis] - score_mat, 0)/threshold[:, np.newaxis]
    return score_mat


def matching(score_mat):

    # Hungarian algorithm to find best matches
    match_rows, match_cols = linear_sum_assignment(-score_mat)
    actually_matched_mask = score_mat[match_rows,
                                      match_cols] > 0 + np.finfo('float').eps
    match_rows = match_rows[actually_matched_mask]
    match_cols = match_cols[actually_matched_mask]
    return (match_rows, match_cols)


class Prediction():
    def __init__(self,
                 frame=None,
                 id=None,
                 position=None,
                 bbox=None,
                 active=False,
                 visible=False,
                 hallucinate=False,
                 age=0,
                 age_visible= 0,
                 memory={}):
        self.active = active
        self.age = age
        self.age_visible = age_visible
        self.memory = memory
        self.outside = True
        self.hallucinate = hallucinate

        self.position = Position(
            position=position, bbox=bbox, frame=frame, is_prediction=True)

        self.visible = visible
        self.id = id

    def set_hallucinate(self, hallucinate):
        self.hallucinate = hallucinate

    def is_outside(self, bbox, img_width, img_height):
        bb_img = np.array([0., 0., img_width, img_height])
        bbox_points = bbox * 1
        bbox_points[2:] += bbox[:2]
        min_ = np.minimum(bbox_points, bb_img)
        max_ = np.maximum(bbox_points, bb_img)
        intersection = np.maximum(
            min_[2] - max_[0], 0) * np.maximum(min_[3] - max_[1], 0)*1.
        overlap_area = intersection / (bbox[2] * bbox[3])
      
        self.outside = (overlap_area < 0.25)
       
        if self.outside:
            self.set_active(False)

    def set_active(self, active):
        self.active = active

    def increase_age(self):
        self.age += 1

    def increase_visible_age(self):
        self.age_visible+= 1

    def reset_age_visible(self):
        self.age_visible = 0

    def set_visible(self, visible, age_update=False):
        self.visible = visible
        
        if not visible:
            self.reset_age_visible()
        else:
            self.increase_visible_age()
        
    def __call__(self):
        return self.position()

    def print(self):

        print("""
            ACTIVE: {}
            ID: {}
            AGE: {}
            AGE_VISIBLE: {}
            POSITION: {}
            VISIBLE: {}""".format(self.active,
                                  self.id,
                                  self.age, self.age_visible,
                                  self.position.print(False),
                                  self.visible))


class Position():

    def __init__(self, position=None, image_position=None, bbox=None, frame=None,
        interacting=None, active=False,
                 interaction_set=None, is_prediction=False, tracker_id=None):

        self.position = position
        self.image_position = image_position
        self.bbox = bbox
        self.frame = frame
        self.interacting = interacting
        self.interaction_set = interaction_set
        self.is_prediction = is_prediction
        self.active = active
        self.tracker_id = tracker_id


    def __call__(self):
        return self.position

    def set_active(self, active):
        self.active = active

    def print(self, show=False):
        msg = """POSITION: {}
                 IMAGE_POSITION: {}
                 BBOX: {}
                 FRAME: {}
                 INTERACTING: {}
                 ACTIVE: {}
            """.format(self.position, self.image_position, self.bbox,
                       self.frame, self.interacting, self.active)
        if show:
            print(msg)
        return msg


class Track():
    def __init__(self, 
            track_id, 
            frame,   
            tracker_id=None, 
            position=None, 
            interacting=False, 
            interaction_set=[], 
            visual_features = None):
        self.age = 0
        self.age_visible = 0
        self.health = 1
        self.active = True
        self.visible = True
        self.logger = defaultdict(list)
        self.gt_id = None
        self.tracker_id = tracker_id
        self.interacting = False
        self.interaction_set = []
        self.occluded = False
        self.has_prediction = False
        self.id = track_id
        self.frame = frame
        self.memory = {}
        self.init_frame = frame
        self.visual_features = visual_features
        self.position = Position(
            position=position, frame=frame, interacting=False, interaction_set=[])
        self.last_position = Position(
            position=position, frame=frame, interacting=interacting, interaction_set=interaction_set)
        self.track_data = {}
        self.prediction_data = {}
        
        self.predictions = {}
        # self.position = position

    def log(self, msg):
        self.logger[self.frame].append(msg)

    def get_trajectory(self):
        trajectory = [[frame, *pos.position]
                      for (frame, pos) in self.track_data.items() if pos.position is not None]
        if len(trajectory) > 0:
            return np.stack(trajectory)
        else:
            return None

    def get_last_position(self, real=True,  prediction_id=0):
        if real:
            if len(self.track_data) > 0:
                return list(self.track_data.values())[-1]

        if self.frame - 1 in self.prediction_data:
            if prediction_id in self.prediction_data[self.frame-1]:
                if self.predictions[prediction_id].position is not None:
                    return self.predictions[prediction_id].position
        return None

    @property
    def active_predictions(self):

        return [pred for pred in self.predictions.values() if (pred.active and pred is not None)]

    def init_prediction(self, id=None, position=None, bbox=None,
                        visible=False, active=True, memory={}, age = 0, age_visible = 0  ):

        
        if id is None:
            if len(self.predictions) == 0:
                id = 0
            else:
                id = max(self.predictions.keys()) + 1
        
        self.predictions[id] = Prediction(frame=copy.deepcopy(self.frame),
                                          id=id,
                                          position=position,
                                          bbox=bbox,
                                          visible=visible,
                                          active=active,
                                          age_visible= age_visible, 
                                          age=int(self.age),
                                          memory=memory)
        self.set_prediction(True)
        return id

    def dehallucinate_predictions(self):
        for key, prediction in self.predictions.items():
            prediction.set_hallucinate(False)

    def set_prediction(self, has_prediction):
        self.has_prediction = has_prediction

    def update_prediction(self, id,
                          position=None,
                          bbox=None,
                          active=None,

                          visible=None):

        prediction = self.get_prediction_id(id)
        self.prediction.frame = self.frame
        for key, item in locals().items():
            if key not in ["id", "self"] and item is not None:
                setattr(prediction, key, item)

    def predict_bb(self, img_width=None, img_height=None):

        # check if bb outside of frame
        assert img_width is not None, "image width not set"
        assert img_height is not None, "image height not set"
        last_position_object = self.get_last_position(
            real=True, prediction_id=0)
        last_image_position = last_position_object.image_position
        if last_image_position is None:
            return
        for pred in self.active_predictions:

            if pred.position.image_position is None:
                pred.position.bbox = np.array([0., 0., 0., 0.])
                continue
            try:
                assert pred.position.frame == self.frame
            except:
              
                for pred in self.active_predictions:
                    pred.print()
                    print("prediction frame", pred.position.frame, self.frame)

                fdf

            predicted_image_position = pred.position.image_position
            try:
                image_translation = predicted_image_position - last_image_position
            except:
                print(predicted_image_position)
                print(last_image_position)
                print(last_position_object.print(True))
            new_bbox = copy.deepcopy(self.get_last_position(real=True).bbox)
            new_bbox[:2] += image_translation
            pred.position.bbox = copy.deepcopy(new_bbox)
            pred.is_outside(new_bbox, img_width, img_height)

    def set_occluded(self, occluded):
        self.occluded = occluded

    def set_bbox(self, bbox):
        self.position.bbox = bbox

    def set_interacting(self, interacting):
        self.interacting = interacting

    def set_interaction_set(self, interaction_set):
        self.interaction_set = interaction_set
        # print(self.interaction_set)

    def get_prediction_id(self, id):
        return self.predictions[id]

    def get_predictions(self):
        
        return list(self.predictions.values())

    def get_bbox_array(self):

        return np.array([self.id, *self.position.bbox])

    def get_position_array(self):
        return np.array([self.id, *self.position()])

    def get_predictions_bbox_array(self, active=True):
        try:
            if active:
                if len(self.active_predictions) == 0:
                    return np.array([])
                return np.stack([[self.id,prediction.id,  self.age, *prediction.position.bbox] for  prediction in (self.active_predictions)])
            else:
                return np.stack([[self.id, prediction.id,  self.age, *prediction.position.bbox] for prediction in (self.active_predictions)])
        except:
            self.print()
            print(self.logger)
            print(self.active_predictions[0].print())
            print(self.frame)
            ssds

    def get_predictions_array(self, active=True):

        if active:
            if len(self.active_predictions) == 0:
                return np.array([])
            return np.stack([[self.id, id_pred, self.age,   *prediction.position()] for id_pred,  prediction in enumerate(self.active_predictions)])
        else:
            return np.stack([[self.id, id_pred, self.age,   *prediction.position()] for id_pred, prediction in enumerate(self.active_predictions)])

    def increase_age(self):
        self.age += 1

    def set_active(self, active):
        self.active = active

    def set_age(self, age):
        self.age = age

    def increase_visible_age(self):
        self.age_visible += 1

    def reset_age_visible(self):
        self.age_visible = 0

    def set_visible(self, visible, age_update=False):
        self.visible = visible
        if not visible:
            self.reset_age_visible()
        else:
            self.increase_visible_age()

    def set_gt(self, gt_id):
        self.gt_id = gt_id

    def reset_predictions(self):
        self.predictions = {}
        
        self.set_prediction(False)

    def reset_prediction_id(self, id):
        self.predictions[id] = Prediction()

    def set_position(self, position=None, image_position=None, bbox=None, frame=None,
                     is_prediction=False, active=False, tracker_id=None):
        self.position = Position(
            position=position, image_position=image_position, bbox=bbox,
            frame=frame, is_prediction=is_prediction, active=active, tracker_id=tracker_id)

    def update_position(self, position=None, interacting=None, bbox=None, interaction_set=None):

        for key, item in locals().items():
            if key not in ["id", "self"] and item is not None:
                setattr(self.position, key, item)

    def update_image_position(self, image_position):
        self.position.frame = self.frame
        self.position.image_position = image_position

    def update_track(self, prediction=False):

        if prediction:
            if self.has_prediction:
                self.position = copy.deepcopy(self.predictions[0].position)

            # self.track_data[self.predictions[0].position.frame] = copy.copy(self.position)

        else:

            if self.position is not None:
                self.track_data[self.frame] = copy.deepcopy(self.position)
            self.prediction_data[self.frame] = copy.deepcopy(
                self.predictions)

    def merge(self, track):
        # assert len(track.track_data) == 0, "Data was already initiated before"

        self.position = copy.deepcopy(track.position)
        self.age = copy.deepcopy(track.age)
        self.age_visible = copy.deepcopy(track.age_visible)
        self.active = copy.deepcopy(track.active)
        self.occluded = copy.deepcopy(track.occluded)
        self.interacting = copy.deepcopy(track.interacting)
        self.tracker_id = copy.deepcopy(track.tracker_id)
        self.set_prediction(False)
        self.reset_predictions()

        self.logger[self.frame].append(
            "Merging {} with {}".format(self.id, track.id))

        track.logger[track.frame].append(
            "Killed by {}: Merging {} with {}".format(self.id, self.id, track.id))

    def set_health(self, health):
        self.health = health

    def print(self):
        print("""Track ID {}:
            AGE: {}                                 HEALTH: {}
            ACTIVE: {}                              GT ID: {}
            POSITION: {}                            AGE_VISIBLE: {}
            LAST POSTION: {}
            FRAME: {}
            INIT FRAME: {}                          OCCLUDED: {}
            TRACKER ID:{}
            TRACK: {}                               HAS_PREDICITON: {}
            """.format(self.id,
                       self.age, self.health, self.active,
                       self.gt_id, self.position.print(
                           False) if self.position else None,
                           self.age_visible,
                       self.last_position.print(
                           False) if self.last_position else None,
                       self.frame,
                       self.init_frame, self.occluded,
                       self.tracker_id, list(self.track_data.keys()),
                       self.has_prediction))

    def kill(self):
        self.logger[self.frame].append(
            "Track killed")
        logger.debug(f"Track {self.id} killed")
        self.set_health(-1)
        self.set_active(False)
        self.tracker_id = None

    def end(self):
        self.logger[self.frame].append(
            "Track Ended")
        self.set_health(0)
        self.set_active(False)
        self.tracker_id = None


class TrackMemory():
    def __init__(self):
        self.log = []

        self.tracker_id_dict = {}

    def setup(self):
        self.memory = {}

    def set_frame(self, frame):
        for track in self.alive_tracks:
            track.frame = frame

    @property
    def occluded_tracks(self):
        return [track for track in self.alive_tracks if track.occluded]

    @property
    def interacting_tracks(self):
        return [track for track in self if track.interacting if track.active]

    def logger_id(self, id):
        track = self.get_track(id)
        print(f"ID: {track.id}, Tracker ID: {track.tracker_id}")
        for frame, msg in track.logger.items():
            print("Frame {}: {}".format(frame, ",".join(msg)))

    def print_logger(self):
        for track in self.memory.values():
            print(f"ID: {track.id} Tracker ID: {track.tracker_id}")
            for frame, info in track.logger.items():
                print(f"{frame}: {info}")

    def increase_age_alive(self):
        [track.increase_age() for track in self.alive_tracks]

    def deactivate_alive(self):
        [track.set_active(False) for track in self.alive_tracks]

    def occlude_alive(self):
        [track.set_occluded(True) for track in self.alive_tracks]

    def new_tracks(self, frame):
        return [track for track in self.alive_tracks if track.init_frame == frame]

    def existing_tracks(self, frame):
        return [track for track in self.alive_tracks if track.init_frame < frame]

    def existing_active_tracks(self, frame):
        return [track for track in self.active_tracks if track.init_frame < frame]

    def existing_alive_tracks(self, frame):
        return [track for track in self.alive_tracks if track.init_frame < frame]

    @property
    def alive_inactive_tracks(self):
        return [track for track in self.alive_tracks if not track.active]

    def step(self):
        for track in self.alive_tracks:
            track.last_position = copy.deepcopy(track.position)
            track.update_track()

    def reset_positions(self):
        for track in self.alive_tracks:
            track.position = None

    def kill(self, track_id):
        self.memory[track_id].set_health(-1)
        self.memory[track_id].set_active(False)

    def end(self, track_id):
        self.memory[track_id].end()

    def initiate(self, track_id, frame,
                 tracker_id=None,  position=None,
                 interacting=False, interaction_set=[], visual_features = None):

        if track_id in self.track_ids:
            track_id = np.max(self.track_ids) + 1
        if visual_features is not None:
            visual_features[:, 1] = track_id
        initiated_track = Track(
            track_id=int(track_id),  tracker_id=tracker_id,
            interacting=interacting, interaction_set=interaction_set,
            frame=frame, position=position,
            visual_features = visual_features )
        initiated_track.log(f"Track initiated ({frame})")
        logger.debug(
            f"Track initiated ({frame}) id: {track_id}, tracker_id: {tracker_id}")
        # if tracker_id is not None:

        self.memory[track_id] = initiated_track

        self.set_tracker_id(tracker_id, track_id)
        return initiated_track, track_id

    def set_gt(self,  track_id, gt_id):
        track = self.get_track(track_id)
        track.set_gt(gt_id)

    @property
    def tracks(self):
        return list(self.memory.values())

    @property
    def track_ids(self):
        return list(self.memory.keys())

    @property
    def tracker_ids(self):
        return [track.tracker_id for track in self.memory.values()]

    def get_track(self, track_id, by_tracker=False):

        if by_tracker:
            track_id = self.tracker_id_dict[track_id]
        return self.memory[track_id]

    def get_prediction(self, track_id, prediction_id):
        return self.memory[track_id].get_prediction_id(prediction_id)

    def reset_predictions(self):
        [track.reset_predictions() for track in self.alive_tracks]

    def reset_predictions_active_tracks(self):
        [track.reset_predictions() for track in self.active_tracks]

    def dehallucinate_predictions(self):
        [track.dehallucinate_predictions() for track in self.alive_tracks]

    @property
    def alive_ids(self):
        return [track.id for track in self.alive_tracks]

    @property
    def alive_tracks(self):
        return [track for track in self.tracks if track.health == 1]

    @property
    def active_ids(self):
        return [track.id for track in self.alive_tracks if track.active]

    @property
    def alive_tracker_ids(self):
        return [track.tracker_id for track in self.alive_tracks if track.health == 1]

    @property
    def active_tracker_ids(self):
        return [track.tracker_id for track in self.alive_tracks if track.active]

    @property
    def active_tracks(self):
        return [track for track in self.alive_tracks if track.active]

    @property
    def valid_tracks(self):
        return [track for track in self.tracks if track.health >= 0]

    def __call__(self):
        for track in self:
            track.print()

    def set_tracker_id(self, tracker_id, track_id):
        """
            ReId right_id into left_id
        """

        if type(tracker_id) == float:
            print("YES it is float", tracker_id, track_id)
            dsm
        if type(track_id) == float:
            print("YES it is float", tracker_id, track_id)
            dsds

        self.tracker_id_dict[tracker_id] = track_id
        logger.debug(f"Set Tracker ID {tracker_id}: {track_id}")
        tracker_id_list = list(self.tracker_id_dict.keys())
        unique_tracker_id = set(tracker_id_list)
        track_id_list = list(self.tracker_id_dict.values())
        unique_track_id = set(track_id_list)
        assert len(tracker_id_list) == len(unique_tracker_id)
        try:
            assert len(track_id_list) == len(unique_track_id)
        except:
            print(self.tracker_id_dict)
            fdfd

    def delete_tracker_id(self, tracker_id):
        if tracker_id in self.tracker_id_dict:
            logger.debug(f"Deleted Tracker ID {tracker_id}")
            del self.tracker_id_dict[tracker_id]

    def deactivate_tracks(self, max_age, max_age_visible):
        for track in self.alive_tracks:
            end = False
            if track.age >= max_age:

                self.delete_tracker_id(track.tracker_id)

                track.end()
                end = True
            if (track.age_visible >= max_age_visible) and not end:
                track.end()
                self.delete_tracker_id(track.tracker_id)
                end = True
            if track.has_prediction and not end:
                
                for pred in track.active_predictions:
                    
                    if track.age_visible > pred.age_visible:
                        track.age_visible  = int(pred.age_visible)
                        print("sth went wrong with the vis age")
                        # print(pred.active)
                        # print(pred.print())
                        # print(pred.__dict__)
                        # print(track.print())
                        # print("END", end)
                        # print(track.age_visible, pred.age_visible, track.frame, track.id, track.visible)
                        # print(track.logger)
                        # ds
                    if pred.age_visible > max_age_visible:
                      
                        pred.set_active(False)
                        # print(track.health)
                        # print("set inactive")
                        # print("age vis pred ", pred.age_visible)
                        # print("active", track.id, track.active)
                        # print("visible", track.age_visible)
                        # print("is_visible", track.visible)
                

    def switch_tracks(self, track_ids):
        track_dict = {}
        targets = []
        sources = []
        tracker_ids = []
        for id1, id2 in track_ids:
            if id1 not in track_dict:
                track_1 = copy.deepcopy(self.get_track(id1))
                track_dict[id1] = track_1
                tracker_ids.append(track_1.tracker_id)
            if id2 not in track_dict:
                track_2 = copy.deepcopy(self.get_track(id2))

                track_dict[id2] = track_2

            targets.append(id2)
            sources.append(id1)

        for id1, id2 in track_ids:

            track_target = self.get_track(id2)
            track_source_real = self.get_track(id1)

            # track_source.print()
            track_source = track_dict[id1]

            # track_source_real.update_track(prediction=True)
            if id1 not in targets:
                track_source_real.tracker_id = np.minimum(
                    -1, np.min(list(self.tracker_id_dict.keys()))) - 1
                track_source_real.set_occluded(True)
                track_source_real.set_active(False)
                track_source_real.position = None
                track_source_real.set_age(1)

            track_target.merge(track_dict[id1])

            if track_dict[id2].tracker_id not in tracker_ids:
                self.delete_tracker_id(track_dict[id2].tracker_id)
            self.set_tracker_id(track_dict[id1].tracker_id, track_target.id)

            # print(len(tracker_ids))
            # tracker_ids.remove(track_dict[id1].tracker_id)
            # print(len(tracker_ids))
            track_target.logger[track_target.frame].append(
                "Including {} in {}".format(id1, id2))
            track_source_real.logger[track_source_real.frame].append(
                "Included by {}".format(id2))

            logger.debug("({}) Including {} ({}) => {} ({})".format(track_target.frame, id1, track_dict[id1].tracker_id,
                                                                    id2, track_dict[id2].tracker_id))

        # for id in tracker_ids:
        #     print(id)
        #     self.delete_tracker_id(id)
            # if id2 == 5:
            #     track_target.print()
            #     print(self.tracker_id_dict)

    def merge_tracks(self,  left_id, right_id):
        """
            - Merging right (right_id) into left (left_id)
            - left (left_id) later is killed
        """
        left_track = self.get_track(left_id)
        right_track = self.get_track(right_id)
        # print("MERGE IDS BEFORE {} {}".format(left_id, right_id))
        # left_track.print()
        # right_track.print()
        # remove left tracker from list, new appearance of tracker will initiate new track

        self.delete_tracker_id(left_track.tracker_id)
        self.delete_tracker_id(right_track.tracker_id)

        self.set_tracker_id(right_track.tracker_id, left_track.id)
        tracker_id_list = list(self.tracker_id_dict.keys())
        unique_tracker_id = set(tracker_id_list)
        track_id_list = list(self.tracker_id_dict.values())
        unique_track_id = set(track_id_list)
        assert len(tracker_id_list) == len(unique_tracker_id)
        assert len(track_id_list) == len(unique_track_id)

        left_track.merge(right_track)
        right_track.kill()

    def get_tracker_id(self, id):
        return self.tracker_id_dict[id]

    def len_active(self, previous=False):
        count = 0
        for track in self.tracks:
            for position in track.track_data.values():

                if position.active:
                    count += 1

        if previous:
            count += len(self.active_tracks)
        return count


class MotionModel():
    def __init__(self, predictor, sequence, tracker):
        self.predictor = predictor
        self.sequence = sequence
        self.tracker = tracker

        img = self.sequence.__getitem__(1, ["rgb"])["rgb"]

        self.img_height, self.img_width, _ = img.shape

        print("""Initializing Trajectory Predictor  \
            sequence: {sequence} \
            model: {model}
            """.format(
            sequence=self.sequence.name,
            model=self.predictor.name))
        self.m = TrackMemory()  # keeping track of ids in scene

    def kill_track(self):
        pass

    # def increase_age(self, track_id):
    #     self.track_memory[].increase_age()
    def project_2d(self):
        pass

    def run(self,
            motion_dim=3,
            reId_metric="L2_APP",
            L2_threshold = 3, 
            IOU_threshold = 0.5, 
            APP_threshold = 0, 
            min_iou_threshold = 0.2, 
            min_appearance_threshold = 1.5, 
            Id_switch_metric="L2",
            visibility=True,
            interactions=False,
            hallucinate=False,
            save_name="",
            save_results=False,
            max_frames=None,
            frames=None,
            clean_transfer = 0, 
            social_interactions=True,
            debug=False,
            max_age=None,
            max_age_visible=None,
            exists_ok=False, 
            

            y0 = None

            ):

           
        if exists_ok == False:
            save_folder = os.path.join(
                "/storage/user/dendorfp/{}/prediction_results/{}_{}/{}/data".format(self.sequence.dataset,
                                                                                    self.predictor.name,
                                                                                    save_name,
                                                                                    self.tracker.name))
            if os.path.exists(os.path.join(save_folder, "{}.txt".format(self.sequence.name))):
                print("File exists")
                return 1 
        self.y0 = y0 if motion_dim == 3 else None
        self.clean_transfer = clean_transfer
        self.L2_threshold = L2_threshold
        self.IOU_threshold = IOU_threshold
        self.min_iou_threshold = min_iou_threshold
        self.min_appearance_threshold = min_appearance_threshold
        self.max_age = max_age
        self.max_age_visible = max_age_visible
        self.debug = debug
        

        self.reId_metric = reId_metric
        self.Id_switch_metric = Id_switch_metric
        self.motion_dim = motion_dim
        if self.motion_dim == 3:
            self.position_row = ["{}_world".format(
                coordinate) for coordinate in ["x", "y", "z"]]
        elif self.motion_dim == 2:
            self.position_row = ["{}_pixel".format(
                coordinate) for coordinate in ["x", "y"]]
            self.visibility = False
        else:
            raise ValueError(
                "`motion_dim` not valid! Value as to be in [2, 3]")
        self.max_frames = max_frames
        self.save_name = save_name
        print("test")

        # settings for run
        self.visibility = visibility
        self.interactions = interactions

        self.m.setup()
        count = 0
        if frames is None:
            frames = self.sequence.frames
        if max_frames is not None:
            max_frames = len(self.sequence.frames)
            frames = self.sequence.frames[:max_frames]

        for frame in tqdm(frames):

            tracks = self.tracker.get_frame(frame)
            if self.visibility:
       
                item = self.sequence.__getitem__(
                    frame, ["map_img", "egomotion"])
                map_dict = item["map_img"]

                visibility = map_dict["visibility"]
              
                if item["egomotion"] is not None:  
                    visibility = np.ones_like(visibility)
                   
                self.visibility_map = visibility 
            # if self.y0 is not None:
            #     H = self.sequence.__getitem__(
            #         frame, ["homography_depth"])["homography_depth"]["IPM"]
            #     self.y0 = get_y0(H)

            egomotion = self.sequence.__getitem__(
                    frame, [ "egomotion"])["egomotion"]
            
            if egomotion is not None:
                H = np.array(self.sequence.__getitem__(
                    frame, [ self.tracker.homography])[self.tracker.homography]["IPM"])
                self.y0 = get_y0(H, self.img_width)
                 
            count += len(tracks)
            # tracks["id"] = tracks["id_x"]

            self.check_tracks(frame, tracks)

            self.predictor(frame, self.m.existing_tracks(frame) if (
                social_interactions or self.predictor.mode == "multiple" or motion_dim == 2) else self.m.occluded_tracks)

            # match with new tracks
            position_list = [track.position for track in self.m.active_tracks]

            [position_list.extend([pred.position for pred in track.get_predictions()])
             for track in self.m.alive_tracks]

            if len(position_list) > 0:
                self.get_image_position(frame, position_list)

            for track in self.m.existing_alive_tracks(frame):
                # try:
                if track.has_prediction:
                    
                    track.predict_bb(img_width=self.img_width, img_height=self.img_height)
                # except:
                #     track.print()
                #     dsd
            if self.visibility and len(self.m.occluded_tracks) > 0 and (self.visibility_map.any()):
                predictions = [
                    track for track in self.m.occluded_tracks if (track.has_prediction and len(track.active_predictions) > 0)]

                if ((len(predictions) > 0) and (len(self.m.active_tracks) > 0)):
                    
                    positions_prediction = np.concatenate(
                                [track.get_predictions_bbox_array() for track in predictions], 0)
                    bbox_target = np.stack(
                                [track.get_bbox_array() for track in self.m.active_tracks], 0)
                    
                    occlusion = overlap_frame(
                        positions_prediction[:, 3:],  bbox_target[:, 1:], img_shape=(self.img_height, self.img_width))
                    # [track.set_visible(False)
                    #                    for track in self.m.occluded_tracks]
                   
                    for track_id  in np.unique(positions_prediction[:, 0]):
                        
                        
                        track_mask = positions_prediction[:, 0] == track_id
                      
                        
                        visible = True
                        for prediction_id,  score in zip( positions_prediction[track_mask, 1], occlusion[track_mask]):
                            
                            pred = self.m.get_prediction(track_id, prediction_id)
                            
                            if score > 0.25:
                                is_visible = False
                            else:
                                is_visible = self.is_visible(pred.position.image_position, img_shape = (self.img_width, self.img_height ))
                          
                                
                            pred.set_visible(is_visible, age_update=True)
                        
                            visible = np.logical_and(visible, is_visible)
        
                        track = self.m.get_track(track_id)
                   
                        track.set_visible(
                            visible)

            if ((len(self.m.new_tracks(frame)) > 0) & (len(self.m.occluded_tracks) > 0)):

                predictions = [
                    track for track in self.m.occluded_tracks if (track.has_prediction and len(track.active_predictions ) > 0 )]
                for pred in predictions:
                    assert pred.active_predictions[0].position.frame == pred.frame, "{} {} ".format(pred.print(), pred.health)
                if len(predictions) > 0:
                    positions_new_detections_valid = [
                            track for track in self.m.new_tracks(frame) if track.position() is not None]
                    if "IOU" in self.reId_metric or self.min_iou_threshold > 0.  or self.motion_dim == 2:
                        positions_prediction = np.concatenate(
                            [track.get_predictions_bbox_array() for track in predictions], 0)
                        
                        image_positions = np.stack((np.clip(positions_prediction[:, 3] + positions_prediction[:, 5]/2, 0, self.img_width -1 ), positions_prediction[:, 4] + positions_prediction[:, 6]), -1)
                        
                        positions_new_detections = np.stack(
                            [track.get_bbox_array() for track in self.m.new_tracks(frame)])

                        score_mat_iou, iou_scores = compute_IOU_scores(
                            positions_prediction[:, 3:], positions_new_detections[:, 1:], threshold=self.IOU_threshold, img_shape = None)
                        score_mat = score_mat_iou*1.
                    if "L2" in self.reId_metric:

                        positions_prediction = np.concatenate(
                            [track.get_predictions_array() for track in predictions], 0)
                       

                        if len(positions_new_detections_valid) > 0:
                            positions_new_detections = np.stack(
                                [track.get_position_array() for track in positions_new_detections_valid])
                            age = positions_prediction[:, 2]
                            score_mat_l2 = compute_L2_scores(
                                positions_prediction[:, 3:-
                                    1], positions_new_detections[:, 1:-1],
                                max_threshold=self.L2_threshold, age=age)
                            score_mat = score_mat_l2
                        else:
                            score_mat = None
                  

                    if self.reId_metric in ["APP", "APP_IOU"] and len(positions_new_detections_valid) > 0 :

                        prediction_appearance = np.concatenate(
                            [track.visual_features for track in predictions])
                        detection_appearance = np.concatenate(
                            [track.visual_features for track in self.m.new_tracks(frame)])
                        
                        appearance_scores = np.ones((len(positions_prediction), 
                            len(positions_new_detections))) * 2
                        # print(prediction_appearance[:, :2], positions_prediction[:,0])
                        # print(detection_appearance[:, :2], positions_new_detections[:, 0])
                        if (len(prediction_appearance) > 0) & (len(detection_appearance)> 0):

                            appearance_mat = compute_dist(prediction_appearance[:, 2:], detection_appearance[:, 2:])
                          
                            # print(appearance_mat.shape, appearance_scores.shape, positions_prediction.shape, positions_new_detections.shape)
                            for k_i , i in enumerate(prediction_appearance[:,1]):
                                for k_j , j in enumerate(detection_appearance[:,1]):
                                    # print(k_j, k_i, i, j,  positions_prediction[:, 0] == i,positions_new_detections[:, 0] == j )
                                    # print(appearance_scores[positions_prediction[:, 0] == i, positions_new_detections[:, 0] == j])
                                    appearance_scores[positions_prediction[:, 0] == i, positions_new_detections[:, 0] == j] = appearance_mat[k_i, k_j]
                        
                        appearance_scores[appearance_scores < self.min_appearance_threshold - np.finfo('float').eps] = 0
                        
                    else: score_mat = None
       
                    if self.reId_metric == "L2_IOU":
                        
                        score_mat = (score_mat_iou + score_mat_l2 ) * ((iou_scores >= self.min_iou_threshold) |  (self.y0[image_positions[:,0].astype(int)] > image_positions[:, 1])[:, np.newaxis]) 
                    elif "L2_APP" in self.reId_metric and len(positions_new_detections_valid) > 0 :

                        prediction_appearance = np.concatenate(
                            [track.visual_features for track in predictions])
                        detection_appearance = np.concatenate(
                            [track.visual_features for track in self.m.new_tracks(frame)])
                        
                        appearance_scores = np.ones((len(positions_prediction), 
                            len(positions_new_detections))) * 2
                        # print(prediction_appearance[:, :2], positions_prediction[:,0])
                        # print(detection_appearance[:, :2], positions_new_detections[:, 0])
                        if (len(prediction_appearance) > 0) & (len(detection_appearance)> 0):

                            appearance_mat = compute_dist(prediction_appearance[:, 2:], detection_appearance[:, 2:])
                          
                            # print(appearance_mat.shape, appearance_scores.shape, positions_prediction.shape, positions_new_detections.shape)
                            for k_i , i in enumerate(prediction_appearance[:,1]):
                                for k_j , j in enumerate(detection_appearance[:,1]):
                                    # print(k_j, k_i, i, j,  positions_prediction[:, 0] == i,positions_new_detections[:, 0] == j )
                                    # print(appearance_scores[positions_prediction[:, 0] == i, positions_new_detections[:, 0] == j])
                                    appearance_scores[positions_prediction[:, 0] == i, positions_new_detections[:, 0] == j] = appearance_mat[k_i, k_j]

                        # print(((2 - appearance_scores ) >= self.min_appearance_threshold), 2 - appearance_scores)

                        # score_mat_iou +
             
                        if self.min_iou_threshold > 0: 
                            iou_mask = ((iou_scores >= self.min_iou_threshold))
                            if self.y0 is not None:
                                iou_mask = np.logical_or(iou_mask , (self.y0[image_positions[:,0].astype(int)] > image_positions[:, 1])[:, np.newaxis]) 
                        else: iou_mask = True
                        
                        if "IOU" in self.reId_metric:       
                            score_mat =  ( score_mat_iou + score_mat_l2 ) *(iou_mask)* ((2 - appearance_scores ) >= self.min_appearance_threshold)
                        else:
                            score_mat =  ( score_mat_l2 ) *(iou_mask)* ((2 - appearance_scores ) >= self.min_appearance_threshold)
                        
                    if self.reId_metric == "APP_IOU":
                        score_mat =  ( score_mat_iou )* ((2 - appearance_scores ) >= self.min_appearance_threshold)
                    elif self.reId_metric == "IOU":
                        score_mat = score_mat_iou
                    elif self.reId_metric == "APP": 
                        score_mat = 2 - appearance_scores

                    if score_mat is not None:
                        unique_ids = np.unique(positions_prediction[:, 0])
                        score_mat_final = np.zeros((len(unique_ids), len(positions_new_detections[:, 0])))
                        for k, pred_id in enumerate(unique_ids):
                           
                            score_mat_final[k] = np.sum(score_mat[positions_prediction[:, 0] == pred_id], 0)

                        row, col = matching(score_mat_final)

                        pred_ids =unique_ids[row]
                        new_det_ids = positions_new_detections[col, 0]

                        for p_id, nd_id in zip(pred_ids, new_det_ids):
                            
                            logger.debug("ReId ({}) {} ({}) => {} ({})".format(
                                frame, nd_id, self.m.get_track(nd_id).tracker_id, p_id, self.m.get_track(p_id).tracker_id))
                            self.m.merge_tracks(p_id, nd_id)
            id_switches = []
            if social_interactions:
                # compute interaction in IOU of new detections and existing tracks

                # 1) get all bounding boxes
                existing_active_tracks = self.m.existing_active_tracks(frame)
                existing_alive_tracks =  [track for track in self.m.existing_alive_tracks(frame) if (track.has_prediction and len(track.active_predictions) > 0 )]

                if len(existing_active_tracks) > 0:
                    predictions = [
                        track for track in self.m.occluded_tracks if (track.has_prediction and len(track.active_predictions) > 0)]
                    target_tracks = [
                        track for track in existing_active_tracks if (track.has_prediction and len(track.active_predictions) > 0 )]
                    if len(target_tracks) > 0:
                        if self.Id_switch_metric == "IOU":

                            bbox_source = np.stack(
                                [track.get_bbox_array() for track in existing_active_tracks if track.position.bbox is not None], 0)

                            source_ids = bbox_source[:, 0]

                            bbox_target = np.concatenate(
                                [track.get_predictions_bbox_array() for track in existing_alive_tracks], 0)
                          
                            score_mat = compute_IOU_scores(
                                bbox_source[:, 1:], 
                                bbox_target[:, 3:], 
                                threshold = 0.7,
                                img_shape = None)
                           
                            target_ids = bbox_target[:, 0]
                            id_prediction = bbox_target[:, 1]
                           
                        if self.Id_switch_metric == "L2":
                            position_source = np.stack(
                                [track.get_position_array() for track in existing_active_tracks if track.position.position is not None], 0)
                            source_ids = position_source[:, 0]

                            if len(target_tracks) > 0:
                                position_target = np.concatenate(
                                    [track.get_predictions_array() for track in existing_alive_tracks if track.has_prediction], 0)

                                target_ids = position_target[:, 0]
                                id_prediction = position_target[:, 1]

                                score_mat = compute_L2_scores(
                                    position_source[:, 1:-1], position_target[:, 3:-1], max_threshold=1)
                            else:
                                score_mat = None
                        if score_mat is not None:
                            row, col = matching(score_mat)
                           
                            unique_target_ids = set(target_ids[col])
                            
                            
                            if len(target_ids[col]) == len(unique_target_ids):
                                id_switches = [(i, j) for (i,  j) in zip(
                                    source_ids[row], target_ids[col]) if i != j]
                            else:

                                target_ids_sorted, id_prediction_sorted, source_ids_sorted = map(
                                    list, zip(*sorted(zip(target_ids[col], id_prediction[col], source_ids[row]))))

                                for (i, j) in zip(
                                    source_ids_sorted, target_ids_sorted):
                                    if ((i != j) & (j not in unique_target_ids)):
                                        id_switches.append((i, j))
                           
                            if len(id_switches) > 0:
                    
                                self.m.switch_tracks(id_switches)

                            # print(ids[row], ids[row])
            if ((len(self.m.new_tracks(frame)) > 0) & (len(self.m.occluded_tracks) > 0) & len(id_switches) > 0):
                score_mat, score_mat_iou, score_mat_l2 = None, None, None
                predictions = [
                    track for track in self.m.occluded_tracks if track.has_prediction]
                # for pred in predictions:
                #     print(pred.print() , pred.active_predictions[0],pred.active_predictions[0].print())
                #     assert pred.active_predictions[0].position.frame == pred.frame
                if len(predictions) > 0:
                    if self.reId_metric in ["IOU", "BOTH"] or self.motion_dim == 2:
                        positions_prediction = np.concatenate(
                            [track.get_predictions_bbox_array() for track in predictions], 0)
                        positions_new_detections = np.stack(
                            [track.get_bbox_array() for track in self.m.new_tracks(frame)])

                        score_mat_iou, iou_scores = compute_IOU_scores(
                            positions_prediction[:, 3:], 
                            positions_new_detections[:, 1:], 
                            threshold=self.IOU_threshold, 
                            img_shape = None)
                        score_mat = score_mat_iou

                    if self.reId_metric in ["L2", "BOTH"]:

                        positions_prediction = np.concatenate(
                            [track.get_predictions_array() for track in predictions], 0)
                        positions_new_detections_valid = [
                            track for track in self.m.new_tracks(frame) if track.position() is not None]

                        if len(positions_new_detections_valid) > 0:
                            positions_new_detections = np.stack(
                                [track.get_position_array() for track in positions_new_detections_valid])
                            age = positions_prediction[:, 2]
                            score_mat_l2 = compute_L2_scores(
                                positions_prediction[:, 3:-
                                    1], positions_new_detections[:, 1:-1],
                                max_threshold=self.L2_threshold, age=age)
                            score_mat = score_mat_l2
                        else:
                            score_mat = None
                    if self.reId_metric == "BOTH":
                        
                        score_mat = (score_mat_iou + score_mat_l2 ) * ((iou_scores >= self.min_iou_threshold) |  (self.y0[image_positions[:,0].astype(int)] > image_positions[:, 1])[:, np.newaxis]) 
                    if score_mat is not None:

                        score_mat_final = np.zeros((len(pred_ids), len(new_det_ids)))
                        for k, pred_id in enumerate(np.unique(positions_prediction[:, 0])):
                            print( np.sum(score_mat[positions_prediction[:, 0] == pred_id], 0).shape)
                            score_mat_final[k] = np.sum(score_mat[positions_prediction[:, 0] == pred_id], 0)

                        row, col = matching(score_mat_final)

                        pred_ids = np.unique(positions_prediction[:, 0])[row, 0]
                        new_det_ids = positions_new_detections[col, 0]
                      
                     
                      
                        for p_id, nd_id in zip(pred_ids, new_det_ids):
                            logger.debug("ReId ({}) {} ({}) => {} ({})".format(
                                frame, nd_id, self.m.get_track(nd_id).tracker_id, p_id, self.m.get_track(p_id).tracker_id))
                            self.m.merge_tracks(p_id, nd_id)

            # add new track if missing
            unique_track_ids = set(tracks.id)
            alive_tracker_ids = set(self.m.alive_tracker_ids)

            # alive_track_ids = set(self.m.alive_ids)

            new_tracks_id = list(
                unique_track_ids.difference(alive_tracker_ids))

            for index, row in tracks[((tracks.id.isin(new_tracks_id)) & (tracks.frame == frame))].iterrows():
                new_track, _ = self.m.initiate(
                    row.id, tracker_id=row.id, frame=frame)

                new_track.set_position(frame=frame,
                                       active=True,
                                       tracker_id=row.id,
                                       is_prediction=False,
                                       bbox=row[["bb_left", "bb_top", "bb_width", "bb_height"]].values)

                if not np.isnan(row[self.position_row[0]]):

                    new_track.update_position(
                        position=row[self.position_row].values)
                new_track.set_age(0)
                new_track.age_visible = 0
                new_track.set_occluded(False)
                new_track.set_active(True)

                self.get_image_position(frame, [new_track.position])

            if hallucinate:

                self.m.dehallucinate_predictions()

                for track in self.m.alive_inactive_tracks:

                    if track.has_prediction and not track.active:
                        prediction = track.get_predictions()[0]

                        # prediction.set_hallucinate(True)

                        if not prediction.visible and prediction.position.bbox is not None and not prediction.outside:
                        # if prediction.position.bbox is not None and not prediction.outside:
                            prediction.set_hallucinate(True)
            self.m.step()
            self.predictor.step(frame, self.m.existing_active_tracks(frame))

            if self.debug:
                assert self.tracker.len(frame) == self.m.len_active(
                ), "lenght tracker {}, length motion model {}".format(self.tracker.len(frame), self.m.len_active())
        self.df = self.create_output_df()

        if save_results:
            save_folder = os.path.join(
                "/storage/user/dendorfp/{}/prediction_results/{}_{}/{}/data".format(self.sequence.dataset,
                                                                                    self.predictor.name,

                                                                                    save_name,
                                                                                    self.tracker.name))
            os.makedirs(save_folder, exist_ok=True)
            self.df.sort_values(["frame", "id"], inplace=True)
            save_file = os.path.join(save_folder, "{}.txt".format(
                self.sequence.name))

            self.df.to_csv(save_file, index=False)
            print("Results successfully save to {}".format(save_file))
            
        return 0

    def get_positions(self, tracks_dict):

        assert positions_prediction.shape[-1] == 5, "Dimension of prediction not correct"
        
        positions_new_detections = [[track.id, *track.position]
                                    for track in tracks_dict["new_tracks"]]

        return np.stack(positions_prediction), np.stack(positions_new_detections)

    def update_logic(self, tracks, trajectory_predictions):
        pass

    def check_tracks(self, frame, tracks):

        self.m.deactivate_tracks(max_age=self.max_age,
                                 max_age_visible=self.max_age_visible)

        # Steps for alive tracks in each time step:
        # 1. Increase age of all alvie tracks
        self.m.increase_age_alive()

        # 2. Set tracks occluded
        self.m.occlude_alive()
        # 3. Deactivate alive tracks
        self.m.deactivate_alive()
        # 4. reset all positions
        self.m.reset_positions()

        # check if track disappeared
        unique_track_ids = set(tracks.id)
        alive_tracker_ids = set(self.m.alive_tracker_ids)

        # alive_track_ids = set(self.m.alive_ids)
        new_tracks = []
        new_tracks_id = list(unique_track_ids.difference(alive_tracker_ids))

        for id in new_tracks_id:
            
            if self.tracker.visual_features is not None:
                visual_features = self.tracker.visual_features[(
                    (self.tracker.visual_features[:, 0] == frame) & 
                     (self.tracker.visual_features[:, 1] == id))]
                
            else: visual_features = None
            
            new_track, _ = self.m.initiate(id, tracker_id=id, frame=frame, visual_features = visual_features)
            new_tracks.append(new_track)

        for index, row in tracks.iterrows():
            t = self.m.get_track(row.id, by_tracker=True)

            t.set_position(frame=frame,
                           active=True,
                           tracker_id=row.id,
                           is_prediction=False,
                           bbox=row[["bb_left", "bb_top", "bb_width", "bb_height"]].values)

            if not np.isnan(row[self.position_row[0]]):
                t.update_position(
                    position=row[self.position_row].values)
            else:
                if t.last_position is not None:
                    if t.last_position.position is not None:
                        t.update_position(
                            position=copy.deepcopy(t.last_position.position))
                # else:
                #     t.update_position(
                #         position=np.array(self.motion_dim))
            # assert t.position.position is not None, "Position cannot be none. Error for id {} in frame {}".format(
            #     t.id, frame)


            # check for id transfer
            if self.clean_transfer > 0 and t.age > 0:
                last_position  = t.get_last_position(
                    real=True, prediction_id=0).positiosan
                new_position = t.position.position
                print( last_position.shape, new_position.shape)
                


            t.set_age(0)
            t.age_visible = 0
            t.set_occluded(False)
            t.set_active(True)
            

            # if row.IOU > 0.25:
            #     # interaciton IOU set to 0.25 !!!!

            #     t.update_position(interacting=True,
            #                       interaction_set=set([self.m.get_track(id, by_tracker=True) for id in row.interaction[:, 0]]))

        # print(alive_tracker_ids.difference(unique_track_ids))
        # print(alive_tracker_ids, unique_track_ids)

        # 5. reset predictions of active tracks
        self.m.reset_predictions_active_tracks()

        self.m.set_frame(frame)

        assert self.tracker.len(frame) == self.m.len_active(previous=True), "Check tracker: length tracker {}, length motion model {} in frame {}".format(
            self.tracker.len(frame), self.m.len_active(previous=True), frame)

        if self.tracker.visual_features is not None:
            for track in self.m.occluded_tracks:
               
                if track.age <= 1: 
                    visual_features= self.tracker.visual_features[(
                        (self.tracker.visual_features[:, 0] == frame - 1) & 
                        (self.tracker.visual_features[:, 1] == track.tracker_id))]
                    visual_features[:, 1 ] = track.id
                    track.visual_features  = visual_features
     
        return

    def get_image_position(self, frame, positions):
        valid_positions = [pos for pos in positions if pos() is not None]

        if len(valid_positions) == 0:
            return

        position_array = np.stack([pos() for pos in valid_positions])

        if self.motion_dim == 3: 
            
            position_array[:, [1, 2]] = position_array[:, [2, 1]]
            if self.tracker.homography:
               
                (p0, p1) = self.sequence.project_homography(
                    position_array[:, 0], position_array[:, 2], frame,  homography = self.tracker.homography, y0 = self.y0)
                z_mask = [True] * len(p0)
            else:
                pose_coordinate = self.sequence.world_to_pose(
                    frame, position_array)
                x = pose_coordinate[:, 0]
                y = pose_coordinate[:, 1]
                z = pose_coordinate[:, 2]
                z_mask = z > 0
                (p0, p1) = self.sequence.project_3d_to_2d(x, y, z)
        else:

            p0 = position_array[:, 0]
            p1 = position_array[:, 1]
            z_mask = [True] * len(p0)

        [setattr(pos, "image_position", np.array([u, v]))
         for u, v, pos, z in zip(p0, p1, valid_positions, z_mask) if z]
        assert len(p0) == len(valid_positions)

    def project_to_bb(self, frame, bottom_points, width=0.7,  height=1.6):
        """
            world_coordinate: bounding boxes (x, y, z) * N (N number of points)
        """
        # world_coordinate

        height_points = bottom_points + height
        points = np.concatenate((bottom_points, height_points), 0)
        points[:, [1, 2]] = points[:, [2, 1]]

        pose_coordinate = self.dataset.data.sequences[0].world_to_pose(
            frame, points)

        pose_coordinate[:len(bottom_points), 0] -= width/2
        pose_coordinate[len(bottom_points):, 0] += width/2

        x = pose_coordinate[:, 0]
        y = pose_coordinate[:, 1]
        z = pose_coordinate[:, 2]
        (p0, p1) = self.dataset.data.sequences[0].project_3d_to_2d(x, y, z)

        bb = np.array([p0[:len(bottom_points)],
                       p1[len(bottom_points):],
                       (p0[len(bottom_points):] - p0[:len(bottom_points)]),
                       p1[:len(bottom_points)] - p1[len(bottom_points):]]).T

        return bb

    def is_visible(self, x, threshold=0, img_shape = None):
        assert img_shape is not None, "image shape not set"
        if not ((0 <= x[0] < img_shape[0] ) and (0 <= x[1] < img_shape[1])): 
            return False 
        
        if self.visibility_map[int(x[1]), int( x[0]) ] == 0:
            return False
        else:
            return True
            
    def create_output_df(self):

        output_list = []

        for track in self.m.valid_tracks:

            for (frame, position) in track.track_data.items():

                if position.active:

                    # print(position.print())
                    if position.image_position is None:
                        image_position = [0, 0]
                    else:
                        image_position = position.image_position
                    bbox = position.bbox
                    if position.position is None:
                        pos = np.zeros(2)
                    else:
                        pos = position.position[:2]

                    output_list.append([frame,
                                            track.id, *bbox,
                                            int(position.active), -1,
                                            int(track.init_frame),
                                            0,
                                            int(position.is_prediction),
                                        *image_position, *pos,  0, 0,1])

            for frame, prediction_dict in track.prediction_data.items():

                # if frame in track.track_data:
                #     if track.track_data[frame].active:
                #         continue

                for key, prediction in prediction_dict.items():
                    if prediction.position.image_position is None:
                        image_position = [0, 0]
                    else:
                        image_position = prediction.position.image_position
                    if prediction.position.bbox is None:
                        bbox = [0, 0, 0, 0]
                    else:
                        bbox = prediction.position.bbox
                    if prediction.position.position is None:
                        position = np.zeros(2)
                    else:
                        position = prediction.position.position[:2]

                    output_list.append([frame, track.id, *bbox, int(prediction.hallucinate), int(prediction.active), -1,
                                        int(prediction.age),
                                        int(1),
                                        *image_position,  *position, key, int(prediction.outside), int(prediction.visible)])
        # 1: frame. 2: id, 3: bb_left, 4: bb_top, 5: bb_width, 6: bb_height
        df = pd.DataFrame(output_list, columns=["frame", "id", "bb_left", "bb_top",
                                                "bb_width", "bb_height", "active", 
                                                "active_prediction", "init_frame", "age",
                                                "is_prediction", "u", "v", "x", "y",
                                                "prediction_id", "outside", "visible"])

        return df

    def load_result(self, save_name):
        save_folder = os.path.join(
            "/storage/user/dendorfp/{}/prediction_results/{}_{}/{}/data".format(self.sequence.dataset,
                                                                                self.predictor.name,
                                                                                save_name,
                                                                                self.tracker.name))
        self.save_name = save_name
        save_file = os.path.join(save_folder, "{}.txt".format(
            self.sequence.name))
        columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                   "active", "active_prediction", "3", "4", "is_prediction", "u", "v", "prediction_id", "is_outside", "visible"]
        self.df = pd.read_csv(save_file)

    def plot_results(self,  show=True, 
            save=False, 
            make_video=False, 
            frames=None, 
            trajectories=True, 
            show_visibility=False , 
            ids = None, 
            analyse = [], 
            save_folder = None):
        self.df.sort_values("frame", inplace = True)
        id_union = list(set(self.df.id.unique()).union(
            set(self.tracker.df.id.unique())))
        from random import randint
        np.random.seed(2)
        color = {}
        n = len(id_union)
        for (id, i) in zip(id_union, range(n)):
            color[id] = ('#%06X' % randint(0, 0xFFFFFF))
        if save_folder is None:
            save_folder = os.path.join("/storage/user/dendorfp/{}/prediction_results/{}_{}/{}/img/{}".format(
                self.sequence.dataset, self.predictor.name, self.save_name, self.tracker.name, self.sequence.name))
        if save:
            try:
                shutil.rmtree(save_folder)
            except:
                pass
            os.makedirs(save_folder, exist_ok=True)

        if frames is None:
            frames = self.sequence.frames[:self.max_frames]

        for frame in tqdm(frames):
            left = self.df[self.df.frame == frame]
            right = self.tracker.get_frame(frame)
            self.plot_frame(frame, left, right, show=show,
                            color=color, save=save_folder if save else None, trajectories=trajectories, show_visibility=show_visibility, ids = ids, analyse = analyse)

        if make_video:
            print("Create Video")

            video_folder = os.path.join(
                "/storage/user/dendorfp/{}/prediction_results/{}_{}/{}/video".format(self.sequence.dataset, self.predictor.name, self.save_name, self.tracker.name))
            os.makedirs(video_folder, exist_ok=True)
            import subprocess
            fps = 20
            subprocess.call(["ffmpeg", "-y", "-r", str(fps), "-start_number","{}".format(np.min(frames)), "-i", "{}/%d.jpg".format(save_folder), "-vcodec",
                            "mpeg4", "-qscale", "5", "-r", str(fps), "{}/{}-{}.mp4".format(video_folder, self.tracker.name,  self.sequence.name)])
            print(f"Save video to {video_folder}")
    def plot_frame(self, frame, left=None, right=None, color=None, show=False,
    save=False, trajectories=True, show_visibility=False, ids = None, analyse  = [] ):

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
            pos[:,1 ] = pos[:, 1] * (1-mask)  +  converted_y * mask
            pos[:,0 ] = pos[:, 0] * (1-mask)  + converted_x * mask
        
            return pos
       
        if not trajectories:
            fig, axes = plt.subplots(1, 2, figsize=(20, 40))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(60, 20))
            
            axes = [axes[0], axes[2], axes[1]]
           
        for ax in axes:
            ax.set_anchor('W')
        for ax in [axes[0], axes[2]]:
            self.sequence.plot_rgb(frame, ax=ax, show=False)
            ax.axis("off")
            ax.set_xlim(0, self.img_width)
            ax.set_ylim(self.img_height, 0)
        axes[1].set_title("BEV ({})".format(frame), fontsize = 30)
        if trajectories or show_visibility:
            
            item = self.sequence.__getitem__(frame, fields=[self.tracker.homography,"map_img", "egomotion"])
            H = np.array(item[self.tracker.homography]["IPM"])
            inv_H = np.array(item[self.tracker.homography]["inv_IPM"])
           

   
            egomotion = item["egomotion"]
           
            
            
            item_img = self.sequence.__getitem__(frame, fields=["map_img", "rgb"])
            ground_mask = item_img["map_img"]["visibility"]

            if egomotion is not None:
                rgb = item_img["rgb"]
                ground_img = np.concatenate((rgb, ground_mask[:, :, np.newaxis]), -1)
                
                offset = np.array(H).dot(np.array([[int(self.img_width/2.), self.img_height, 1]]).T).T
                offset = offset[:, :-1]/offset[:, -1:]
                y0 = get_y0(H, self.img_width)
            else: 
                y0 = self.y0               
                ground_img = item["map_img"]["rgb"]
            mask_shape = ground_img.shape 
            pixels = np.array(list(itertools.product(range(mask_shape[0]), range(mask_shape[1]))))
            

            pixel_positions = pixels[ground_mask.reshape(-1) != 0 ]
            pixel_positions = pixel_positions[:, (1, 0)]
            
            pos = H.dot(np.concatenate((pixel_positions, np.ones((len(pixel_positions), 1))), -1).T).T
            pos = pos/pos[:, -1:]
           

            
            if y0 is not None:
                new_pos = pix2real(H, pos, pixel_positions,  y0, img_width = self.img_width)
            else: new_pos = pos
            
            
            new_pos = new_pos * 10
            mins = np.min(new_pos, 0)


            new_pos[:, 0 ]-=mins[0]
            new_pos[:, 1 ]-=mins[1]
            maxs =  np.around(np.max(new_pos, 0))
            
            img_mask = np.zeros((int(maxs[1] + 1),int( maxs[0] + 1) , 4))
            
            new_pos = np.around(new_pos).astype(int)
            
            img_mask[new_pos[:, 1], new_pos[:, 0]] = ground_img[pixel_positions[:, 1], pixel_positions[:, 0]]/255.
            img_mask[..., :3] = ndimage.median_filter(img_mask[..., :3], size=2)
            img_mask[..., -1:] = ndimage.maximum_filter(img_mask[..., -1:], size=2)
            
            axes[1].imshow(img_mask)

            # axes[1].invert_xaxis()
            # axes[1].invert_yaxis()
            axes[1].axis( "equal")
            
            axes[1].set_ylim(0, maxs[1])
            axes[1].set_xlim(0, maxs[0])
            axes[1].axis("off")
      

             

            
            scale = 0.1
            origin = -mins[:2]  # - np.array(visibility["offset"])

                # p = patches.Polygon([[(p[0]/scale + origin[0], (p[1]/scale + origin[1])]
                #     for p in visibility["polygon"]],  facecolor='red', alpha=0.2)
                # axes[2].add_patch(p)
                
        text_ids = []
        for index, row in left.iterrows():
            if ids is not None:
                if row.id not in ids: 
                    continue
            axes[0].set_title("QuoVadis ({})".format(frame), fontsize = 30)
            id_color = color[int(row.id)]
          
            if row.active == 1:
                facecolor = "none"
                if len(analyse) > 0: 
                    for key in analyse:
                        if len(key[(key.frame == row.frame) & ((key.tracker_id == row.id))]) > 0: 
                            alpha = 0.5
                            faceolor = "red"
                        else:
                            alpha = 0.3
                            

                else: 
                    alpha = 1.
                linewidth = 2
                if row.is_prediction == 1:
                    linestyle = "--"
                    marker = "^"
                else:
                    linestyle = "-"
                    marker = "."
            else:
                if row.age > 0 and row.active_prediction == 1:
                    facecolor = id_color
                    marker = "^"
                    if len(analyse) > 0: 
                        for key in analyse:
                            
                            if len(key[(key.frame == row.frame) & ((key.tracker_id == row.id))]) > 0: 
                                
                                print("is in key")
                                alpha =  0.8
                                facecolor = "green"
                            else:
                                alpha  = 0.2 
                                facecolor = "black"

                    else: 
                        alpha = 0.15
                    if row.visible == 0:
                        linestyle = None
                        linewidth = 0
                    else: 
                        linestyle="--"
                        linewidth = 1
                    
                else:
                    continue
            if ((0 < row["u"] < self.img_width) and int(row.id) not in text_ids):
                axes[0].text(x = row["u"], y = np.minimum(row["v"], 1000), s = "{}".format(int(row.id)))
                text_ids.append(int(row.id))
            
            axes[0].plot(row["u"], row["v"], marker, color=color[int(row.id)])
            
            rect = patches.Rectangle((row["bb_left"], row["bb_top"]), row["bb_width"], row["bb_height"],
                                     linewidth=linewidth, edgecolor=color[int(row.id)], facecolor=facecolor, 
                                     linestyle=linestyle, alpha=alpha)
            axes[0].add_patch(rect)
           
            if trajectories:
                if ((row["x"] == 0) and (row["y"] == 0)):
                    continue
                x = row["x"]
                y = row["y"]

                if egomotion is not None:
                    x+= egomotion["median"][0] + offset[:, 0]
                    y+= egomotion["median"][1] + offset[:, 1]
                pos_x = x/scale + origin[0]
                pos_y = y/scale + origin[1]
                # pos_x = row["x"]
                # pos_y = row["y"]
                
                if row.outside == 1.:

                    marker = "p"
                elif row.active == 0.:
                    marker = "^"
                else:
                    if row.is_prediction == 1:
                        marker = "*"
                    else:
                        marker = "."
                axes[1].scatter(pos_x, pos_y, s=1000, edgecolors='black',
                                marker=marker, color=color[int(row.id)])
               
                if row.active == 0:
                    
                    circle = patches.Circle((pos_x, pos_y),L2_threshold(row.age,max_threshold=self.L2_threshold)/scale, facecolor = color[int(row.id)], alpha = 0.3)
                    axes[1].add_patch(circle)
                    track = self.m.get_track(row.id)

                    # old_traj  = self.df[(self.df.frame < frame) & (self.df.id == row.id)][["x", "y"]].values
                    # if egomotion is not None:
                    #         print(egomotion["median"])
                    #         old_traj[:, :2]+= egomotion["median"]
                    # axes[1].plot(old_traj[:, 0]/scale + origin[0] , old_traj[:, 1]/scale + origin[1], "-", alpha = 0.3 , linewidth = 2,   color = color[int(row.id)])
                    for pred in track.prediction_data[row.frame].values():
                        try:
                            trajectory = pred.memory["trajectory"][:, :2]
                            
                            if egomotion is not None:
                                trajectory_plot= trajectory + egomotion["median"] + offset
                            else: trajectory_plot = trajectory
                            axes[1].plot(trajectory_plot[:, 0]/scale + origin[0] , trajectory_plot[:, 1]/scale + origin[1], "--", alpha = 0.5 , linewidth = 5,   color = color[int(row.id)])
                            if self.motion_dim == 3:
                                (p0, p1) = self.sequence.project_homography(
                                    trajectory[:, 0], trajectory[:, 1], frame,  homography = self.tracker.homography, y0 = y0)
                                axes[0].plot(p0 , p1,"--", alpha = 0.3 ,    color = color[int(row.id)])
                            else: 
                                p0 = trajectory_plot[:, 0]
                                p1= trajectory_plot[:, 1]
                            
                                axes[0].plot(p0 , p1,"--", alpha = 1. ,    color = color[int(row.id)])
                        except:
                            print(traceback.print_exc())
                            pass

        for index, row in right.iterrows():
            axes[2].set_title("{} ({})".format(self.tracker.name, frame), fontsize = 30)
            rect = patches.Rectangle((row["bb_left"], row["bb_top"]), row["bb_width"], row["bb_height"],
                                        linewidth=2, edgecolor=color[int(row.id)], facecolor='none')
            axes[2].add_patch(rect)
        if show:
            plt.show()
        if save:

            self.df.sort_values(["frame", "id"], inplace=True)
            save_file = os.path.join(save, "{}.jpg".format(
                frame))

            plt.savefig(save_file, bbox_inches='tight', )

            plt.clf()
            plt.cla()
            plt.cla()

        return fig, ax

    def run_eval(self):
        model = "{}_{}".format(self.predictor.name,self.save_name)
        command = f"cd /usr/wiss/dendorfp/dvl/projects/TrackingMOT/TrackEval; bash eval_bash_scripts/{self.sequence.dataset}_model.sh {self.sequence.name} {self.tracker.name} {model}"
        os.system(command)

    def run_analysis(self): 

        model = "{}_{}".format(self.predictor.name,self.save_name)
        output_path = "/storage/user/dendorfp/{}/prediction_results/{}/{}/output/{}/{}".format(self.sequence.dataset, model, self.tracker.name, self.sequence.name, self.tracker.name)
        data_path = "/storage/user/dendorfp/{}/prediction_results/{}/{}/data/".format(self.sequence.dataset,model, self.tracker.name)


        baseline_output_path = "/storage/user/dendorfp/{}/tracker/{}/output/{}/{}".format( self.sequence.dataset, self.tracker.name, self.sequence.name, self.tracker.name)
        baseline_data_path = "/storage/user/dendorfp/{}/tracker/{}/data".format( self.sequence.dataset,self.tracker.name, self.tracker.name)
        sequence = self.sequence.name
        idsw_path = os.path.join(output_path, "idsw", "{}.txt".format(sequence))
        idtr_path = os.path.join(output_path, "idtr", "{}.txt".format(sequence))
        matches_path = os.path.join(output_path, "matches", "{}.txt".format(sequence))
        prediction_path = os.path.join(data_path, "{}.txt".format(sequence))

        baseline_idsw_path = os.path.join(baseline_output_path, "idsw", "{}.txt".format(sequence))
        baseline_matches_path = os.path.join(baseline_output_path, "matches", "{}.txt".format(sequence))
        baseline_path = os.path.join(baseline_data_path, "{}.txt".format(sequence))
        baseline_idtr_path = os.path.join(baseline_output_path, "idtr", "{}.txt".format(sequence))
        print(baseline_idtr_path)
        # load predictions
        prediction = pd.read_csv(prediction_path)
        prediction = prediction[prediction.active == 1]
        prediction_idsw = pd.read_csv(idsw_path)    
        prediction_idsw["prediction"] = 1
        

        prediction_idtr = pd.read_csv(idtr_path)
        
        prediction_idtr["prediction"] = 1
        prediction_matches = pd.read_csv(matches_path)



        baseline_idsw = pd.read_csv(baseline_idsw_path)
        baseline_idsw["baseline"] = 1
        baseline_matches = pd.read_csv(baseline_matches_path)
        baseline_data = pd.read_csv(baseline_path, 
                                names = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "x", "y", "z", "a"])
        baseline_idtr = pd.read_csv(baseline_idtr_path)

        baseline_idtr["baseline"] = 1
        assert baseline_idtr_path != idtr_path

        baseline = baseline_idsw.merge(baseline_matches, left_on = ["frame", "tracker_id"], right_on = ["frame", "tracker_id"], how = "left")
        baseline_data = baseline_data.merge(baseline_matches[["frame", "tracker_id", "gt_id"]], left_on = ["frame", "id"],right_on = ["frame", "tracker_id"], how="left")

        baseline_idsw = baseline_idsw.merge(baseline_matches, left_on=["frame", "tracker_id"], right_on=["frame", "tracker_id"], how="left")
        baseline_idsw["gap"] = None

        for index, row in baseline_idsw.iterrows(): 
            before_df = baseline_data[((baseline_data.id == row.prev_tracker_id) & (baseline_data.frame < row.frame) & 
                                    (baseline_data.gt_id == row.gt_id))]
            idx_last = before_df["frame"] == before_df['frame'].max()
            last_row = before_df[idx_last]
            baseline_idsw.loc[((baseline_idsw.frame == row.frame ) &
                            (baseline_idsw.tracker_id == row.tracker_id)),"gap"] = row.frame - last_row.frame.item()
        for index, row in baseline_idtr.iterrows(): 
            before_df = baseline_data[((baseline_data.id == row.tracker_id) & (baseline_data.frame < row.frame) & 
                                    (baseline_data.gt_id == row.prev_gt_id))]
            idx_last = before_df["frame"] == before_df['frame'].max()
            last_row = before_df[idx_last]
            
            baseline_idtr.loc[((baseline_idtr.frame == row.frame ) &
                            (baseline_idtr.tracker_id == row.tracker_id)),"gap"] = row.frame - last_row.frame.item()


        prediction_idsw = prediction_idsw.merge(prediction_matches, left_on=["frame", "tracker_id"], right_on=["frame", "tracker_id"], how="left")
        prediction_idsw["gap"] = None


        
        prediction = prediction.merge(prediction_matches[["frame", "tracker_id", "gt_id"]], left_on = ["frame", "id"],right_on = ["frame", "tracker_id"], how="left")
        for index, row in prediction_idsw.iterrows(): 
            before_df = prediction[((prediction.id == row.prev_tracker_id) & (prediction.frame < row.frame) & 
                                    (prediction.gt_id == row.gt_id))]
            idx_last = before_df["frame"] == before_df['frame'].max()
            last_row = before_df[idx_last]
            
            prediction_idsw.loc[((prediction_idsw.frame == row.frame ) &
                            (prediction_idsw.tracker_id == row.tracker_id)),"gap"] = row.frame - last_row.frame.item()
        # prediction_result = prediction_idsw.merge(prediction_matches, left_on = ["frame", "tracker_id"], right_on = ["frame", "tracker_id"], how = "left")

        for index, row in prediction_idtr.iterrows():
            before_df = prediction[((prediction.id == row.tracker_id) & (prediction.frame < row.frame) & 
                                    (prediction.gt_id == row.prev_gt_id))]
            idx_last = before_df["frame"] == before_df['frame'].max()
            last_row = before_df[idx_last]
            
            prediction_idtr.loc[((prediction_idtr.frame == row.frame ) &
                            (prediction_idtr.tracker_id == row.tracker_id)),"gap"] = row.frame - last_row.frame.item()

        results = prediction_idsw[["frame", "gt_id", "prediction", "gap", "tracker_id"]].merge(baseline_idsw[["frame", "gt_id", "baseline", "gap"]], left_on =["frame", "gt_id"], right_on=["frame", "gt_id"], how = "outer")
        results.fillna(0,
         inplace = True)
        results["gap"] = np.maximum(results["gap_x"].values, results["gap_y"].values)
        
        TP = results[((results.prediction == 0 ) & (results.baseline == 1))]
        FN = results[((results.prediction == 1 ) & (results.baseline == 1))]
        FP = results[((results.prediction == 1 ) & (results.baseline == 0))]
        
        results_idtr = prediction_idtr[["frame", "prev_gt_id", "gt_id", "prediction", "gap", "tracker_id"]].merge(
            baseline_idtr[["frame", "prev_gt_id", "gt_id", "baseline", "gap"]], left_on =["frame", "prev_gt_id", "gt_id"], right_on=["frame", "prev_gt_id", "gt_id"], how = "outer")
        results_idtr.fillna(0, inplace = True)
        results_idtr["gap"] = np.maximum(results_idtr["gap_x"].values, results_idtr["gap_y"].values)
        results_idtr.frame-=1
        TP_idtr = results_idtr[((results_idtr.prediction == 0 ) & (results_idtr.baseline == 1))]
        FN_idtr = results_idtr[((results_idtr.prediction == 1 ) & (results_idtr.baseline == 1))]
        FP_idtr = results_idtr[((results_idtr.prediction == 1 ) & (results_idtr.baseline == 0))]
        
        self.analyse = Namespace(TP = TP , FN = FN , FP = FP, TP_idtr = TP_idtr, FP_idtr = FP_idtr, FN_idtr = FN_idtr)
