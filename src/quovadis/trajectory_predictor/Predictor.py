import copy
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .kalman_filter import KalmanFilter


def get_predictor(cfg):
    if cfg.PRED_MODEL == "linear":
        predictor = LinearPredictor()
    elif cfg.PRED_MODEL == "kalman":
        predictor = KalmanFilterPredictor(dt=1/30.,
                                          measurement_uncertainty_x=0.1,
                                          measurement_uncertainty_y=0.1,
                                          process_uncertainty=0.1)
    elif cfg.PRED_MODEL == "static":
        predictor = StaticPredictor()
    elif cfg.PRED_MODEL == "mggan":
        
        predictor = MGGANPredictor(
            model_path=cfg.MGGAN_WEIGHTS,
            device=cfg.DEVICE,
            nr_predictions=cfg.NR_PREDICTIONS,
            dataset_name="motsynth",
            pred_len=15,
            dt=cfg.DT
        )
    elif cfg.PRED_MODEL == "gan":
        predictor = MGGANPredictor(
            model_path=cfg.MGGAN_WEIGHTS,
            device=cfg.DEVICE,
            nr_predictions=cfg.NR_PREDICTIONS,
            dataset_name="motsynth",
            pred_len=15,
            dt=cfg.DT
        )
    else:
        raise ValueError(
            "No valid prediction model given for option 'PRED_MODEL'.")
    return predictor


class Predictor(ABC):
   
    def __init__(self, sequence=None):
        self.sequence = sequence
        
    def __call__(self, frame, tracks,  sequence=None):
        if len(tracks) == 0:
            return
        if self.mode == "single":
            for track in tracks:
                predictions = self.predict(frame=frame, tracks=track,
                                           sequence=sequence)
                if predictions is not None:

                    for key, prediction in predictions.items():

                        track.init_prediction(id=key, **prediction)
                else:
                    track.set_prediction(False)
        elif self.mode == "multiple":
            predictions = self.predict(frame, tracks=tracks, sequence=sequence)
            for track in tracks:
                track_predictions = predictions[track.id]

                if track_predictions is None:
                    track.set_prediction(False)
                else:
                    for key, prediction in track_predictions.items():
                        if prediction is None:
                            continue
                        track.init_prediction(id=key, **prediction)

    @abstractmethod
    def predict(self, frame, tracks, sequence):
        pass

    def step(self, frame=None, tracks=None):
        pass

    @abstractmethod
    def predict_trajectory(self, **kwargs):
        pass


class KalmanFilterPredictor(Predictor):
    name = "kalman_filter"
    mode = "single"

    def __init__(self, dt=1/20., process_uncertainty=1,
                 measurement_uncertainty_x=2,
                 measurement_uncertainty_y=2):
        self.dt = dt
        self.process_uncertainty = process_uncertainty
        self.measurement_uncertainty_x = measurement_uncertainty_x
        self.measurement_uncertainty_y = measurement_uncertainty_y

    def predict_trajectory(self, obs, gap=1, max_back_view=20,  **kwargs):

        obs_xy = obs[["x", "y"]].values
        initial_state = np.concatenate(
            (obs_xy[0, :2], np.array([0.,  0.])))
        kf = KalmanFilter(initial_state=initial_state,
                          measurement_uncertainty_x=self.measurement_uncertainty_x,
                          measurement_uncertainty_y=self.measurement_uncertainty_y,
                          process_uncertainty=self.process_uncertainty,
                          )
        if len(obs_xy) > 1:
            kf.smooth(obs_xy[1:])

        return kf.predictSequence(time=range(gap))

    def predict(self, track, frame, **kwargs):

        last_position_object = track.get_last_position(
            real=True, prediction_id=0)
        last_position = last_position_object.position
        if last_position is None:
            return None
        if "kf" in track.memory:
            kf = track.memory["kf"]
        else:
            initial_state = np.concatenate(
                (last_position[:2], np.array([0.,  0.])))
            kf = KalmanFilter(initial_state=initial_state,
                              measurement_uncertainty_x=self.measurement_uncertainty_x,
                              measurement_uncertainty_y=self.measurement_uncertainty_y,
                              process_uncertainty=self.process_uncertainty,
                              frame=frame)
            track.memory["kf"] = kf
        predict_kf = copy.deepcopy(kf)
        frames = np.arange(frame, frame + 180)
        trajectory = predict_kf.predictSequence(frames)

        prediction = kf.predict(frame=frame)

        if len(last_position) == 3:
            new_position = np.concatenate(
                (prediction[0, :2], last_position[-1:]))
        elif len(last_position) == 2:
            new_position = prediction[0]
        if len(track.active_predictions) > 0:

            age_visible = track.active_predictions[0].age_visible

        else:
            age_visible = 0

        return {0: {"position": new_position, "age_visible": age_visible,
                    "memory": {"frames": frames, "trajectory": trajectory}}}

    def step(self, frame=None, tracks=None):

        for track in tracks:

            current_position = track.position()
            if current_position is None:
                continue
            assert frame == track.position.frame
            assert track.occluded == False

            if "kf" in track.memory:
                kf = track.memory["kf"]
            else:
                initial_state = np.concatenate(
                    (current_position[:2], np.array([0.,  0.])))
                kf = KalmanFilter(initial_state=initial_state,
                                  measurement_uncertainty_x=self.measurement_uncertainty_x,
                                  measurement_uncertainty_y=self.measurement_uncertainty_y,
                                  process_uncertainty=self.process_uncertainty,
                                  frame=frame)
                track.memory["kf"] = kf

            try:

                kf.step(current_position[:2], frame=frame)
            except:
                track.print()
                print(kf)
                traceback.print_exc()


class OraclePredictor(Predictor):
    name = "oracle"
    mode = "single"

    def __init__(self, tracker=None,
                 sequence=None,
                 motion_dim=3):
        self.motion_dim = motion_dim
        self.tracker = tracker
        if self.motion_dim == 3:
            self.position_row = ["{}_world".format(
                coordinate) for coordinate in ["x", "y", "z"]]
        elif self.motion_dim == 2:
            self.position_row = ["{}_pixel".format(
                coordinate) for coordinate in ["x", "y"]]
        super().__init__(sequence)

    def predict(self, frame, track, **kwargs):

        last_position_object = track.get_last_position(
            real=True, prediction_id=0)

        label_row = self.get_labels(
            frame, last_position_object.frame, last_position_object.tracker_id)

        if len(label_row) > 0:

            new_position = label_row[self.position_row].values[0]
            return {0: {"position": new_position}}

        else:
            return None

    def get_labels(self, frame, last_frame, last_track_id):

        labels = self.sequence.__getitem__(frame, ["labels"])["labels"]

        gt_id = self.tracker.df[((self.tracker.df.id == last_track_id)
                                 & (self.tracker.df.frame == last_frame))]["gt_id"].item()
        if gt_id > 0:
            return labels[labels.id == int(gt_id)]
        else:
            return []


class TheilSenPredictor(Predictor):

    name = "theil_sen"
    mode = "single"

    def predict(self, track, frame, **kwargs):
        memory = {}
        last_position_object = track.get_last_position(
            real=True, prediction_id=0)
        last_frame = last_position_object.frame
        last_position = last_position_object.position
        dt = frame - last_frame
        if last_position is None:
            return None
        if track.has_prediction:

            prediction = track.get_predictions()[0]

            if "v" in prediction.memory:
                v = prediction.memory["v"]
            else:
                past_traj = track.get_trajectory()
                if past_traj is None:
                    return None
                v = self.estimate_velocity(past_traj)
        else:
            past_traj = track.get_trajectory()
            if past_traj is None:
                return None
            v = self.estimate_velocity(past_traj)
            if v is None:
                return None
            else:
                memory["v"] = v

        new_position = last_position + v * dt

        return {0: {"position": new_position, "memory": memory}}

    def predict_trajectory(self, obs, gap=1, max_back_view=20,  **kwargs):
        from sklearn.linear_model import TheilSenRegressor

        estimators = [TheilSenRegressor(
            random_state=42), TheilSenRegressor(random_state=42)]
        obs_xy = obs[["x", "y"]].values
        if max_back_view is not None:
            obs_xy = obs_xy[-max_back_view:]

        if len(obs_xy) == 1:
            return np.repeat(obs_xy, gap, axis=0)

        time = np.arange(-len(obs_xy) + 1, 1)[:, np.newaxis]

        prediction = []
        time_prediction = np.arange(1, gap+1)
        assert len(time) == len(obs_xy), "{} {}".format(len(time), len(obs_xy))
        for index, estimator in enumerate(estimators):
            try:
                estimator.fit(time, obs_xy[:, index])
            except:
                print(obs_xy)
                dsds
            pred = estimator.predict(time_prediction[:, np.newaxis])
            prediction.append(pred)

        return np.stack(prediction, 0).T

    def estimate_velocity(self, past_traj, max_back_view=20):
        if len(past_traj) < 2:
            return None
        past_traj = past_traj[-max_back_view:]
        dt = past_traj[1:, :1] - past_traj[:-1, :1]
        assert (dt != 0).all(), "dt cannot be 0"
        dx = past_traj[1:, 1:] - past_traj[:-1, 1:]
        v = dx/dt
        v_mean = np.mean(v, 0)
        v_mean[-1] = 0
        return v_mean

    def get_labels(self, frame):

        labels = self.sequence.__getitem__(frame, ["labels"])["labels"]

        return labels


class LinearPredictor(Predictor):
    name = "linear"
    mode = "single"

    def predict(self, track, frame, **kwargs):
        memory = {}
        last_position_object = track.get_last_position(
            real=True, prediction_id=0)
        last_frame = last_position_object.frame
        last_position = last_position_object.position
        dt = frame - last_frame
        if last_position is None:
            return None
        if track.has_prediction:

            prediction = track.get_predictions()[0]

            if "v" in prediction.memory:
                v = prediction.memory["v"]
            else:
                past_traj = track.get_trajectory()
                if past_traj is None:
                    return None
                v = self.estimate_velocity(past_traj)
        else:
            past_traj = track.get_trajectory()
            if past_traj is None:
                return None
            v = self.estimate_velocity(past_traj)
            if v is None:
                return None
            else:
                memory["v"] = v

        new_position = last_position + v * dt

        return {0: {"position": new_position, "memory": memory}}

    def predict_trajectory(self, obs, gap=1, max_back_view=20,  **kwargs):

        obs_xy = obs[["x", "y"]].values
        past_traj = obs_xy[-max_back_view:]
        if len(obs_xy) == 1:
            v = np.array([0., 0.])
        else:
            v = past_traj[1:] - past_traj[:-1]

            v = np.mean(v, 0)
        v = v[np.newaxis, :]

        t = np.arange(1, gap + 1)
        dx = v * t[:, np.newaxis]

        return obs_xy[-1:] + dx

    def estimate_velocity(self, past_traj, max_back_view=20):
        if len(past_traj) < 2:
            return None
        past_traj = past_traj[-max_back_view:]
        dt = past_traj[1:, :1] - past_traj[:-1, :1]
        assert (dt != 0).all(), "dt cannot be 0"
        dx = past_traj[1:, 1:] - past_traj[:-1, 1:]
        v = dx/dt
        v_mean = np.mean(v, 0)
        v_mean[-1] = 0
        return v_mean

    def get_labels(self, frame):

        labels = self.sequence.__getitem__(frame, ["labels"])["labels"]

        return labels


class MultimodalLinearPredictor(LinearPredictor):
    name = "multimodal_linear"
    mode = "single"

    def __init__(self, sequence, nr_predictions=3, alpha=30, motion_dim=3):
        super().__init__()
        self.motion_dim = motion_dim
        self.sequence = sequence
        self.nr_predictions = nr_predictions
        self.alpha = alpha
        self.angles_rad = np.radians(
            np.linspace(-self.alpha, self.alpha, self.nr_predictions))
        rotation_matrices_list = []
        self.angles_rad[0], self.angles_rad[1] = self.angles_rad[1], self.angles_rad[0]
        for phi in self.angles_rad:
            if self.motion_dim == 3:
                rotation_matrices_list.append(np.array([[np.cos(phi), -np.sin(phi), 0],
                                                        [np.sin(phi), np.cos(
                                                            phi), 0],
                                                        [0, 0, 1]]))
            else:
                rotation_matrices_list.append(np.array([[np.cos(phi), -np.sin(phi)],
                                                        [np.sin(phi), np.cos(
                                                            phi)]]))

        self.rot_mats = np.stack(rotation_matrices_list)

    def predict(self, track, frame, **kwargs):

        memory = {}
        last_position_object = track.get_last_position(
            real=True, prediction_id=0)
        last_frame = last_position_object.frame
        last_position = last_position_object.position
        dt = frame - last_frame
        if last_position is None:
            return None
        if track.has_prediction:
            prediction = track.get_predictions()[0]
            memory = prediction.memory
            if "v" in memory:
                v = memory["v"]
                new_predictions = {}
                for k, v_i in enumerate(v):
                    new_predictions[k] = {
                        "position": last_position + v_i * dt, "memory": memory}
                return new_predictions

        past_traj = track.get_trajectory()
        if past_traj is None:
            return None
        v = self.estimate_velocity(past_traj)
        if v is None:
            return None
        v = v[np.newaxis, :, np.newaxis]
        v = (self.rot_mats @ v)[:, :, 0]
        memory["v"] = v
        v_norm = np.sqrt(np.sum(v**2, axis=1))

        assert (np.sum(v_norm) == 0) or ((abs(np.sum(
            abs(v_norm / (v_norm[0] + 1e-16))) - self.nr_predictions)) < 1e-5), f"{v_norm}"
        new_predictions = {}
        for k, v_i in enumerate(v):
            new_predictions[k] = {
                "position": last_position + v_i * dt, "memory": memory}

        return new_predictions


class StaticPredictor(Predictor):
    name = "static"
    mode = "single"

    def predict(self, track, **kwargs):

        last_position_object = track.get_last_position(
            real=True, prediction_id=0)

        last_position = last_position_object.position

        if last_position is None:
            return None

        if len(track.active_predictions) > 0:

            age_visible = track.active_predictions[0].age_visible

        else:
            age_visible = 0

        return {0: {"position":  copy.deepcopy(last_position), "age_visible": age_visible}}

    def predict_trajectory(self, obs, gap=1, **kwargs):

        obs_xy = obs[["x", "y"]].values[-1:]

        return np.repeat(obs_xy, gap, axis=0)


class LSTMPredictor(Predictor):
    name = "lstm"
    mode = "single"

    def predict(self, track, **kwargs):
        return copy.deepcopy(track.last_position.position)


class GANPredictor(Predictor):

    def predict(self):
        pass

    def get_batch(self, frame, tracks, **kwargs):
        data_list = []
        last_position_dict = {}
        for track in tracks:
            past_traj = track.get_trajectory()

            if past_traj is None:
                continue
            last_position_object = track.get_last_position(
                real=True, prediction_id=0)
            last_position = last_position_object.position
            if last_position is None:
                continue
            last_position_dict[track.id] = last_position

            ids = np.ones(len(past_traj)) * track.id
            data_list.append(np.concatenate(
                (ids[:, np.newaxis], past_traj[:, :-1]), 1))

        if len(data_list) == 0:

            return None
        data = np.concatenate(data_list, 0)

        data = data[:, (1, 0, 2, 3)]

        batch = self.dataset.create_scene(data, frame - 1)

        if batch is None:

            return None
        return batch, last_position_dict


class MGGANPredictor(GANPredictor):

    mode = "multiple"
    datasets = ["mot16", "motsynth", "mot17"]

    def __init__(self,
                 model_path,
                 device,
                 img_min=np.array([10, 10]),
                 checkpoint="best",
                 dataset_name="motsynth",
                 sequence="001",
                 nr_predictions=3,
                 strategy="uniform_expected",
                 pred_len=12,
                 dt=0.2,
                 ):

        super().__init__()
        assert strategy in (
            "uniform_expected",
            "sampling",
            "expected",
            "rejection",
            "smart_expected",
            "smart_sampling",
            "uniform_sampling",
        )

        assert dataset_name in self.datasets, f"`dataset_name`: {dataset_name} not valid"
        print(
            f"Starting MGGAN: nr_predictions: {nr_predictions}, model: {model_path}")
        self.strategy = strategy
        self.name = "GAN" if "1gen" in model_path else "MGGAN"
        self.img_min = img_min
        self.sequence = sequence
        self.nr_predictions = nr_predictions
        self.dataset_name = dataset_name

        from mggan.data_utils import OnlineDataset
        from mggan.model.train import PiNetMultiGeneratorGAN  # noqa: E2

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            map_location = torch.device('cuda:0')
        else:
            map_location = torch.device('cpu')
        model, config = PiNetMultiGeneratorGAN.load_from_path(
            Path(model_path), checkpoint, map_location=map_location)
        model.G.to(device)

        model.device = device
        model.G.eval()
        self.predictor = model

        self.dataset = OnlineDataset(
            img_min=self.img_min,
            cnn=(config.grid_size > 0),
            grid_size_in_global=config.grid_size,
            grid_size_out_global=config.grid_size,
            scene_batching=True,
            goal_gan=False,
            local_features=False,
            scaling_global=0.5,
            load_semantic_map=config.load_semantic_map,
            time_step=dt,
            pred_len=pred_len,
            obs_len=config.obs_len,
            dataset_name=self.dataset_name,
            sequence=self.sequence)
        self.frames_per_step = int(
            self.dataset.framerate * self.dataset.time_step)
        self.interpolation_steps = np.ones(
            self.frames_per_step)/self.frames_per_step

        self.predictor.G.set_pred_len(pred_len)
        self.predictor.G.pred_len = pred_len

    def predict_trajectory(self, obs, gap=1, **kwargs):

        frame = obs["frame"].max()
        x = obs[["frame", "id", "x", "y"]].values

        batch = self.dataset.create_scene(x, frame, padding=False)

        pred_len = int(
            np.ceil(gap / (self.dataset.framerate * self.dataset.time_step)))

        self.predictor.G.set_pred_len(pred_len)
        self.predictor.G.pred_len = pred_len

        prediction = self.predictor.get_predictions_batch(
            batch, num_preds=self.nr_predictions, strategy=self.strategy)
        print(prediction)
        out_dxdy = prediction["out_dxdy"].unsqueeze(1).cpu().numpy()
        # print("predictions", out_dxdy[:, :, 0])
        interpolated_dxdy = out_dxdy * \
            self.interpolation_steps[np.newaxis, :,
                                     np.newaxis, np.newaxis, np.newaxis]

        time_steps, int_step, nr_pred, nr_ped,  dim = interpolated_dxdy.shape
        interpolated_dxdy = np.reshape(
            interpolated_dxdy, (time_steps * int_step, nr_pred, nr_ped,  dim))
        interpolated_xy = np.cumsum(interpolated_dxdy, 0)
        last_position = x[x[:, 0] == frame][0, -2:]

        predicted_traj = interpolated_xy[:, :, 0] + \
            last_position[np.newaxis, np.newaxis, :2]

        return predicted_traj[: gap]

    def predict(self, frame, tracks, **kwargs):
        batch_out = self.get_batch(frame, tracks)

        if batch_out is None:
            return {track.id: None for track in tracks}
        (batch, last_position_dict) = batch_out
        prediction = self.predictor.get_predictions_batch(
            batch, num_preds=self.nr_predictions, strategy=self.strategy)

        out_dxdy = prediction["out_dxdy"].unsqueeze(1).cpu().numpy()
        # print("predictions", out_dxdy[:, :, 0])
        interpolated_dxdy = out_dxdy * \
            self.interpolation_steps[np.newaxis, :,
                                     np.newaxis, np.newaxis, np.newaxis]

        time_steps, int_step, nr_pred, nr_ped,  dim = interpolated_dxdy.shape
        interpolated_dxdy = np.reshape(
            interpolated_dxdy, (time_steps * int_step, nr_pred, nr_ped,  dim))
        interpolated_xy = np.cumsum(interpolated_dxdy, 0)

        ids = list(batch["ids"])
        output = {}
        frames = np.arange(len(interpolated_dxdy)) + frame
        track_ids = []

        for track in tracks:
            assert track.id not in track_ids
            track_ids.append(track.id)
            if track.id not in ids and not track.has_prediction:
                output[track.id] = None
            elif track.has_prediction:
                # update current prediction
                for j in range(self.nr_predictions):
                    output[track.id] = {}

                    for key, prediction in track.predictions.items():

                        if not prediction.active:

                            output[track.id][key] = None

                        else:
                            memory = prediction.memory
                            predicted_traj = memory["trajectory"][memory["frames"] == frame][0]
                            output[track.id][key] = {
                                "position": predicted_traj, "memory": memory, "age_visible": prediction.age_visible}
                          
            elif track.id in ids:
                id_prediction = ids.index(track.id)
                last_position = last_position_dict[track.id]
                predicted_traj = interpolated_xy[:, :, id_prediction] + \
                    last_position[np.newaxis, np.newaxis, :2]
                predicted_traj = np.concatenate((predicted_traj, np.ones_like(
                    predicted_traj)[:, :, :1] * last_position[-1]), -1)

                output[track.id] = {}

                for j in range(self.nr_predictions):
                    traj = predicted_traj[0, j]
                    assert traj.shape == (3,), traj.shape

                    output[track.id][j] = {"position": traj, "memory":
                                           {"frames": frames,
                                            "trajectory": predicted_traj[:, j]}
                                           }
            else:
                output[track.id] == None
        return output


if __name__ == "__main__":
    import pandas as pd
    obs = np.stack((np.arange(10), np.arange(10))).T
    obs = pd.DataFrame(obs, columns=["x", "y"])
    TS = TheilSenPredictor()
    x = TS.predict_trajectory(obs, 9)
