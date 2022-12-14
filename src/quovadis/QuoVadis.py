import copy
import itertools
import logging
import os
import shutil
import traceback
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from tqdm import tqdm

from quovadis.datasets.utils import pix2real

from .utils import L2_threshold, compute_IOU_scores, compute_L2_scores, compute_dist, matching, overlap_frame, get_y0


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()


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
                 age_visible=0,
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
        self.age_visible += 1

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
                 visual_features=None):
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
                        visible=False, active=True, memory={}, age=0, age_visible=0):

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
                                          age_visible=age_visible,
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
                return np.stack([[self.id, prediction.id,  self.age, *prediction.position.bbox] for prediction in (self.active_predictions)])
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

        else:

            if self.position is not None:
                self.track_data[self.frame] = copy.deepcopy(self.position)
            self.prediction_data[self.frame] = copy.deepcopy(
                self.predictions)

    def merge(self, track):
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
                 interacting=False, interaction_set=[], visual_features=None):

        if track_id in self.track_ids:
            track_id = np.max(self.track_ids) + 1
        if visual_features is not None:
            visual_features[:, 1] = track_id
        initiated_track = Track(
            track_id=int(track_id),  tracker_id=tracker_id,
            interacting=interacting, interaction_set=interaction_set,
            frame=frame, position=position,
            visual_features=visual_features)
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
                        track.age_visible = int(pred.age_visible)
                        print("sth went wrong with the vis age")

                    if pred.age_visible > max_age_visible:

                        pred.set_active(False)

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

            track_target.logger[track_target.frame].append(
                "Including {} in {}".format(id1, id2))
            track_source_real.logger[track_source_real.frame].append(
                "Included by {}".format(id2))

            logger.debug("({}) Including {} ({}) => {} ({})".format(track_target.frame, id1, track_dict[id1].tracker_id,
                                                                    id2, track_dict[id2].tracker_id))

    def merge_tracks(self,  left_id, right_id):
        """
            - Merging right (right_id) into left (left_id)
            - left (left_id) later is killed
        """
        left_track = self.get_track(left_id)
        right_track = self.get_track(right_id)

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


class QuoVadis():
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

    def run(self,
            motion_dim=3,
            reId_metric="L2_APP",
            L2_threshold=3,
            IOU_threshold=0.5,
            min_iou_threshold=0.2,
            min_appearance_threshold=1.5,
            Id_switch_metric="L2",
            visibility=True,
            save_directory=None,
            save_results=False,
            max_frames=None,
            frames=None,
            clean_transfer=0,
            debug=False,
            max_age=None,
            max_age_visible=None,
            y0=None

            ):

        self.y0 = y0 if motion_dim == 3 else None
        self.clean_transfer = clean_transfer
        self.L2_threshold = L2_threshold
        self.IOU_threshold = IOU_threshold
        self.min_iou_threshold = min_iou_threshold
        self.min_appearance_threshold = min_appearance_threshold
        self.max_age = max_age
        self.max_age_visible = max_age_visible
        self.debug = debug
        self.save_directory = save_directory
        self.tracker_folder = os.path.join(
            self.save_directory, self.sequence.dataset)
        self.save_folder = os.path.join(self.tracker_folder, self.tracker.name)
        self.results_folder = os.path.join(self.save_folder, "data")
        self.configs_folder = os.path.join(self.save_folder, "cfgs")
        self.vis_save_folder = os.path.join(self.save_folder, "vis")

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

        # settings for run
        self.visibility = visibility

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
                    frame, ["map", "egomotion"])
                map_dict = item["map"]

                visibility = map_dict["visibility"]
                if item["egomotion"] is not None:
                    visibility = np.ones_like(visibility)

                self.visibility_map = visibility

            egomotion = self.sequence.__getitem__(
                frame, ["egomotion"])["egomotion"]

            if egomotion is not None:
                H = np.array(self.sequence.__getitem__(
                    frame, [self.tracker.homography])[self.tracker.homography]["IPM"])
                self.y0 = get_y0(H, self.img_width)

            count += len(tracks)

            self.check_tracks(frame, tracks)
            self.predictor(frame, self.m.existing_tracks(frame) if (
                self.predictor.mode == "multiple" or motion_dim == 2) else self.m.occluded_tracks)

            # match with new tracks
            position_list = [track.position for track in self.m.active_tracks]

            [position_list.extend([pred.position for pred in track.get_predictions()])
             for track in self.m.alive_tracks]

            if len(position_list) > 0:
                self.get_image_position(frame, position_list)

            for track in self.m.existing_alive_tracks(frame):
                # try:
                if track.has_prediction:

                    track.predict_bb(img_width=self.img_width,
                                     img_height=self.img_height)

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

                    for track_id in np.unique(positions_prediction[:, 0]):

                        track_mask = positions_prediction[:, 0] == track_id

                        visible = True
                        for prediction_id,  score in zip(positions_prediction[track_mask, 1], occlusion[track_mask]):

                            pred = self.m.get_prediction(
                                track_id, prediction_id)

                            if score > 0.25:
                                is_visible = False
                            else:
                                is_visible = self.is_visible(
                                    pred.position.image_position, img_shape=(self.img_width, self.img_height))

                            pred.set_visible(is_visible, age_update=True)

                            visible = np.logical_and(visible, is_visible)

                        track = self.m.get_track(track_id)

                        track.set_visible(
                            visible)

            if ((len(self.m.new_tracks(frame)) > 0) & (len(self.m.occluded_tracks) > 0)):

                predictions = [
                    track for track in self.m.occluded_tracks if (track.has_prediction and len(track.active_predictions) > 0)]
                for pred in predictions:
                    assert pred.active_predictions[0].position.frame == pred.frame, "{} {} ".format(
                        pred.print(), pred.health)
                if len(predictions) > 0:
                    positions_new_detections_valid = [
                        track for track in self.m.new_tracks(frame) if track.position() is not None]
                    if "IOU" in self.reId_metric or self.min_iou_threshold > 0. or self.motion_dim == 2:
                        positions_prediction = np.concatenate(
                            [track.get_predictions_bbox_array() for track in predictions], 0)

                        image_positions = np.stack((np.clip(
                            positions_prediction[:, 3] + positions_prediction[:, 5]/2, 0, self.img_width - 1), positions_prediction[:, 4] + positions_prediction[:, 6]), -1)

                        positions_new_detections = np.stack(
                            [track.get_bbox_array() for track in self.m.new_tracks(frame)])

                        score_mat_iou, iou_scores = compute_IOU_scores(
                            positions_prediction[:, 3:], positions_new_detections[:, 1:], threshold=self.IOU_threshold, img_shape=None)
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

                    if self.reId_metric in ["APP", "APP_IOU"] and len(positions_new_detections_valid) > 0:

                        prediction_appearance = np.concatenate(
                            [track.visual_features for track in predictions])
                        detection_appearance = np.concatenate(
                            [track.visual_features for track in self.m.new_tracks(frame)])

                        appearance_scores = np.ones((len(positions_prediction),
                                                     len(positions_new_detections))) * 2

                        if (len(prediction_appearance) > 0) & (len(detection_appearance) > 0):

                            appearance_mat = compute_dist(
                                prediction_appearance[:, 2:], detection_appearance[:, 2:])

                            for k_i, i in enumerate(prediction_appearance[:, 1]):
                                for k_j, j in enumerate(detection_appearance[:, 1]):

                                    appearance_scores[positions_prediction[:, 0] == i,
                                                      positions_new_detections[:, 0] == j] = appearance_mat[k_i, k_j]

                        appearance_scores[appearance_scores <
                                          self.min_appearance_threshold - np.finfo('float').eps] = 0

                    else:
                        score_mat = None

                    if self.reId_metric == "L2_IOU":

                        score_mat = (score_mat_iou + score_mat_l2) * ((iou_scores >= self.min_iou_threshold) | (
                            self.y0[image_positions[:, 0].astype(int)] > image_positions[:, 1])[:, np.newaxis])
                    elif "L2_APP" in self.reId_metric and len(positions_new_detections_valid) > 0:

                        prediction_appearance = np.concatenate(
                            [track.visual_features for track in predictions])
                        detection_appearance = np.concatenate(
                            [track.visual_features for track in self.m.new_tracks(frame)])

                        appearance_scores = np.ones((len(positions_prediction),
                                                     len(positions_new_detections))) * 2

                        if (len(prediction_appearance) > 0) & (len(detection_appearance) > 0):

                            appearance_mat = compute_dist(
                                prediction_appearance[:, 2:], detection_appearance[:, 2:])

                            for k_i, i in enumerate(prediction_appearance[:, 1]):
                                for k_j, j in enumerate(detection_appearance[:, 1]):

                                    appearance_scores[positions_prediction[:, 0] == i,
                                                      positions_new_detections[:, 0] == j] = appearance_mat[k_i, k_j]

                        if self.min_iou_threshold > 0:
                            iou_mask = ((iou_scores >= self.min_iou_threshold))
                            if self.y0 is not None:
                                iou_mask = np.logical_or(iou_mask, (self.y0[image_positions[:, 0].astype(
                                    int)] > image_positions[:, 1])[:, np.newaxis])
                        else:
                            iou_mask = True

                        if "IOU" in self.reId_metric:
                            score_mat = (score_mat_iou + score_mat_l2) * (iou_mask) * \
                                ((2 - appearance_scores) >=
                                 self.min_appearance_threshold)
                        else:
                            score_mat = (score_mat_l2) * (iou_mask) * \
                                ((2 - appearance_scores) >=
                                 self.min_appearance_threshold)

                    if self.reId_metric == "APP_IOU":
                        score_mat = (
                            score_mat_iou) * ((2 - appearance_scores) >= self.min_appearance_threshold)
                    elif self.reId_metric == "IOU":
                        score_mat = score_mat_iou
                    elif self.reId_metric == "APP":
                        score_mat = 2 - appearance_scores

                    if score_mat is not None:
                        unique_ids = np.unique(positions_prediction[:, 0])
                        score_mat_final = np.zeros(
                            (len(unique_ids), len(positions_new_detections[:, 0])))
                        for k, pred_id in enumerate(unique_ids):

                            score_mat_final[k] = np.sum(
                                score_mat[positions_prediction[:, 0] == pred_id], 0)

                        row, col = matching(score_mat_final)

                        pred_ids = unique_ids[row]
                        new_det_ids = positions_new_detections[col, 0]

                        for p_id, nd_id in zip(pred_ids, new_det_ids):

                            logger.debug("ReId ({}) {} ({}) => {} ({})".format(
                                frame, nd_id, self.m.get_track(nd_id).tracker_id, p_id, self.m.get_track(p_id).tracker_id))
                            self.m.merge_tracks(p_id, nd_id)
            id_switches = []

            if ((len(self.m.new_tracks(frame)) > 0) & (len(self.m.occluded_tracks) > 0) & len(id_switches) > 0):
                score_mat, score_mat_iou, score_mat_l2 = None, None, None
                predictions = [
                    track for track in self.m.occluded_tracks if track.has_prediction]

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
                            img_shape=None)
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

                        score_mat = (score_mat_iou + score_mat_l2) * ((iou_scores >= self.min_iou_threshold) | (
                            self.y0[image_positions[:, 0].astype(int)] > image_positions[:, 1])[:, np.newaxis])
                    if score_mat is not None:

                        score_mat_final = np.zeros(
                            (len(pred_ids), len(new_det_ids)))
                        for k, pred_id in enumerate(np.unique(positions_prediction[:, 0])):
                            print(
                                np.sum(score_mat[positions_prediction[:, 0] == pred_id], 0).shape)
                            score_mat_final[k] = np.sum(
                                score_mat[positions_prediction[:, 0] == pred_id], 0)

                        row, col = matching(score_mat_final)

                        pred_ids = np.unique(
                            positions_prediction[:, 0])[row, 0]
                        new_det_ids = positions_new_detections[col, 0]

                        for p_id, nd_id in zip(pred_ids, new_det_ids):
                            logger.debug("ReId ({}) {} ({}) => {} ({})".format(
                                frame, nd_id, self.m.get_track(nd_id).tracker_id, p_id, self.m.get_track(p_id).tracker_id))
                            self.m.merge_tracks(p_id, nd_id)

            # add new track if missing
            unique_track_ids = set(tracks.id)
            alive_tracker_ids = set(self.m.alive_tracker_ids)

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

            self.m.step()
            self.predictor.step(frame, self.m.existing_active_tracks(frame))

            if self.debug:
                assert self.tracker.len(frame) == self.m.len_active(
                ), "lenght tracker {}, length motion model {}".format(self.tracker.len(frame), self.m.len_active())
        self.df = self.create_output_df()

        if save_results:
            print(self.sequence.name)

            os.makedirs(self.save_folder, exist_ok=True)
            os.makedirs(self.results_folder, exist_ok=True)
            os.makedirs(self.configs_folder, exist_ok=True)

            self.df.sort_values(["frame", "id"], inplace=True)
            self.save_file = os.path.join(self.results_folder, "{}.txt".format(
                self.sequence.name))

            self.df.to_csv(self.save_file, index=False)
            print("Results successfully save to {}".format(self.save_file))

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

            else:
                visual_features = None

            new_track, _ = self.m.initiate(
                id, tracker_id=id, frame=frame, visual_features=visual_features)
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

            # check for id transfer
            if self.clean_transfer > 0 and t.age > 0:
                last_position = t.get_last_position(
                    real=True, prediction_id=0).positiosan
                new_position = t.position.position
                print(last_position.shape, new_position.shape)

            t.set_age(0)
            t.age_visible = 0
            t.set_occluded(False)
            t.set_active(True)

        # 5. reset predictions of active tracks
        self.m.reset_predictions_active_tracks()

        self.m.set_frame(frame)

        assert self.tracker.len(frame) == self.m.len_active(previous=True), "Check tracker: length tracker {}, length motion model {} in frame {}".format(
            self.tracker.len(frame), self.m.len_active(previous=True), frame)

        if self.tracker.visual_features is not None:
            for track in self.m.occluded_tracks:

                if track.age <= 1:
                    visual_features = self.tracker.visual_features[(
                        (self.tracker.visual_features[:, 0] == frame - 1) &
                        (self.tracker.visual_features[:, 1] == track.tracker_id))]
                    visual_features[:, 1] = track.id
                    track.visual_features = visual_features

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
                    position_array[:, 0], position_array[:, 2], frame,  homography=self.tracker.homography, y0=self.y0)
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

    def is_visible(self, x, threshold=0, img_shape=None):
        assert img_shape is not None, "image shape not set"
        if not ((0 <= x[0] < img_shape[0]) and (0 <= x[1] < img_shape[1])):
            return False

        if self.visibility_map[int(x[1]), int(x[0])] == 0:
            return False
        else:
            return True

    def create_output_df(self):
        output_list = []
        for track in self.m.valid_tracks:
            for (frame, position) in track.track_data.items():
                if position.active:

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
                                        *image_position, *pos,  0, 0, 1])

            for frame, prediction_dict in track.prediction_data.items():
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
        df = pd.DataFrame(output_list, columns=["frame", "id", "bb_left", "bb_top",
                                                "bb_width", "bb_height", "active",
                                                "active_prediction", "init_frame", "age",
                                                "is_prediction", "u", "v", "x", "y",
                                                "prediction_id", "outside", "visible"])

        return df

    def load_result(self, save_folder):

        save_file = os.path.join(save_folder, "{}.txt".format(
            self.sequence.name))
        columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
                   "active", "active_prediction", "3", "4", "is_prediction", "u", "v", "prediction_id", "is_outside", "visible"]
        self.df = pd.read_csv(save_file)

    def vis_results(self,
                    show=True,
                    save=False,
                    make_video=False,
                    frames=None,
                    trajectories=True,
                    show_visibility=False,
                    ids=None,


                    save_folder=None):

        self.df.sort_values("frame", inplace=True)
        id_union = list(set(self.df.id.unique()).union(
            set(self.tracker.df.id.unique())))
        from random import randint
        np.random.seed(2)
        color = {}
        n = len(id_union)

        for (id, i) in zip(id_union, range(n)):
            color[id] = ('#%06X' % randint(0, 0xFFFFFF))
        if save:
            if self.vis_save_folder is None:
                raise ValueError("Missing input 'save_folder'")

            try:
                shutil.rmtree(self.vis_save_folder)
            except:
                pass
            os.makedirs(self.vis_save_folder, exist_ok=True)

        if frames is None:
            frames = self.sequence.frames[:self.max_frames]

        for frame in tqdm(frames):

            preds = self.df[self.df.frame == frame]

            self.plot_frame(frame, preds, show=show,
                            color=color,
                            save=self.vis_save_folder if save else None,
                            trajectories=trajectories, show_visibility=show_visibility, ids=ids)

        if make_video:
            print("Creating Video")
            video_folder = os.path.join(self.vis_save_folder, 'video')
            image_folder = os.path.join(self.vis_save_folder, 'images')
            os.makedirs(video_folder, exist_ok=True)
            import subprocess
            fps = 20
            subprocess.call(["ffmpeg", "-y", "-r", str(fps), "-start_number", "{}".format(np.min(frames)), "-i", "{}/%d.jpg".format(image_folder), "-vcodec",
                            "mpeg4", "-qscale", "5", "-r", str(fps), "{}/{}-{}.mp4".format(video_folder, self.tracker.name,  self.sequence.name)])

            print(f"Save video to {video_folder}")

            subprocess.call(["ffmpeg", "-i",
                             "{}/{}-{}.mp4".format(video_folder,
                                                   self.tracker.name,  self.sequence.name),
                             "-vf", "fps=20,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                             "-loop",  "0",  "{}/{}-{}.gif".format(video_folder, self.tracker.name,  self.sequence.name)])

            print(f"Save gif to {video_folder}")

    def plot_frame(self, frame, preds, color=None, show=False,
                   save=False, trajectories=True, show_visibility=False, ids=None,

                   ):
        scale = 0.1
        if not trajectories:
            fig, ax = plt.subplots(1)
            axT = None

        else:
            fig, (ax, axT) = plt.subplots(1, 2, figsize=(40, 20))

        ax.set_anchor('W')
        if axT:
            axT.set_anchor('W')

        self.sequence.plot_rgb(frame, ax, show=False)
        ax.axis("off")
        ax.set_xlim(0, self.img_width)
        ax.set_ylim(self.img_height, 0)

        if (trajectories or show_visibility):

            axT.set_title("BEV ({})".format(frame), fontsize=30)

            item = self.sequence.__getitem__(
                frame, fields=["rgb", "homography", "map", "egomotion"])
            H = np.array(item[self.tracker.homography]["IPM"])

            egomotion = item["egomotion"]
            ground_mask = item["map"]["visibility"]

            rgb = item["rgb"]
            if egomotion is not None:

                ground_img = np.concatenate(
                    (rgb, ground_mask[:, :, np.newaxis]), -1)

                offset = np.array(H).dot(
                    np.array([[int(self.img_width/2.), self.img_height, 1]]).T).T
                offset = offset[:, :-1]/offset[:, -1:]
                y0 = get_y0(H, self.img_width)
            else:
                y0 = self.y0
                ground_img = item["map"]["rgb"]

            mask_shape = rgb.shape
            pixels = np.array(list(itertools.product(
                range(mask_shape[0]), range(mask_shape[1]))))

            pixel_positions = pixels[ground_mask.reshape(-1) != 0]
            pixel_positions = pixel_positions[:, (1, 0)]

            pos = H.dot(np.concatenate(
                (pixel_positions, np.ones((len(pixel_positions), 1))), -1).T).T
            pos = pos/pos[:, -1:]

            if y0 is not None:
                new_pos = pix2real(H, pos, pixel_positions,
                                   y0, img_width=self.img_width)
            else:
                new_pos = pos

            new_pos = new_pos * 10
            mins = np.min(new_pos, 0)

            new_pos[:, 0] -= mins[0]
            new_pos[:, 1] -= mins[1]
            maxs = np.around(np.max(new_pos, 0))

            img_mask = np.zeros((int(maxs[1] + 1), int(maxs[0] + 1), 4))

            new_pos = np.around(new_pos).astype(int)

            img_mask[new_pos[:, 1], new_pos[:, 0]
                     ] = ground_img[pixel_positions[:, 1], pixel_positions[:, 0]]/255.
            img_mask[..., :3] = ndimage.median_filter(
                img_mask[..., :3], size=2)
            img_mask[..., -
                     1:] = ndimage.maximum_filter(img_mask[..., -1:], size=2)

            axT.axis("equal")

            axT.set_ylim(0, maxs[1])
            axT.set_xlim(0, maxs[0])

            axT.imshow(img_mask)

            axT.axis("off")

            origin = -mins[:2]
        item = self.sequence.__getitem__(
            frame, fields=["rgb", "homography", "map", "egomotion"])
        H = np.array(item[self.tracker.homography]["IPM"])
        #     inv_H = np.array(item[self.tracker.homography]["inv_IPM"])

        egomotion = item["egomotion"]
        text_ids = []
        for index, row in preds.iterrows():

            if ids is not None:
                if row.id not in ids:

                    continue
            ax.set_title("QuoVadis ({})".format(frame), fontsize=30)
            id_color = color[int(row.id)]

            if row.active == 1:
                facecolor = "none"

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

                    alpha = 0.15
                    if row.visible == 0:
                        linestyle = None
                        linewidth = 0
                    else:
                        linestyle = "--"
                        linewidth = 1

                else:
                    continue
            if ((0 < row["u"] < self.img_width) and int(row.id) not in text_ids):
                ax.text(x=row["u"], y=np.minimum(row["v"], 1000),
                        s="{}".format(int(row.id)))
                text_ids.append(int(row.id))

            ax.plot(row["u"], row["v"], marker, color=color[int(row.id)])

            rect = patches.Rectangle((row["bb_left"], row["bb_top"]), row["bb_width"], row["bb_height"],
                                     linewidth=linewidth, edgecolor=color[int(
                                         row.id)], facecolor=facecolor,
                                     linestyle=linestyle, alpha=alpha)
            ax.add_patch(rect)

            if trajectories:
                if ((row["x"] == 0) and (row["y"] == 0)):
                    continue
                x = row["x"]
                y = row["y"]

                if egomotion is not None:
                    x += egomotion["median"][0] + offset[:, 0]
                    y += egomotion["median"][1] + offset[:, 1]
                pos_x = x/scale + origin[0]
                pos_y = y/scale + origin[1]

                if row.outside == 1.:

                    marker = "p"
                elif row.active == 0.:
                    marker = "^"
                else:
                    if row.is_prediction == 1:
                        marker = "*"
                    else:
                        marker = "."
                axT.scatter(pos_x, pos_y, s=1000, edgecolors='black',
                            marker=marker, color=color[int(row.id)])

                if row.active == 0:

                    circle = patches.Circle((pos_x, pos_y), L2_threshold(
                        row.age, max_threshold=self.L2_threshold)/scale, facecolor=color[int(row.id)], alpha=0.3)
                    axT.add_patch(circle)
                    track = self.m.get_track(row.id)

                    for pred in track.prediction_data[row.frame].values():
                        try:
                            trajectory = pred.memory["trajectory"][:, :2]

                            if egomotion is not None:
                                trajectory_plot = trajectory + \
                                    egomotion["median"] + offset
                            else:
                                trajectory_plot = trajectory
                            axT.plot(trajectory_plot[:, 0]/scale + origin[0], trajectory_plot[:, 1] /
                                     scale + origin[1], "--", alpha=0.5, linewidth=5,   color=color[int(row.id)])
                            if self.motion_dim == 3:
                                (p0, p1) = self.sequence.project_homography(
                                    trajectory[:, 0], trajectory[:, 1], frame,  homography=self.tracker.homography, y0=y0)
                                ax.plot(p0, p1, "--", alpha=0.3,
                                        color=color[int(row.id)])
                            else:
                                p0 = trajectory_plot[:, 0]
                                p1 = trajectory_plot[:, 1]

                                ax.plot(p0, p1, "--", alpha=1.,
                                        color=color[int(row.id)])
                        except:
                            print(traceback.print_exc())
                            pass

        if show:
            plt.show()

        if save:
            os.makedirs(os.path.join(save, "images"), exist_ok=True)
            self.df.sort_values(["frame", "id"], inplace=True)
            save_file = os.path.join(save, "images", "{}.jpg".format(
                frame))

            plt.savefig(save_file, bbox_inches='tight', )

            plt.clf()
            plt.cla()
            plt.close()

        return fig, ax
