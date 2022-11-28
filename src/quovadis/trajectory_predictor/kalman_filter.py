import numpy as np
from filterpy.kalman import KalmanFilter as KF


class KalmanFilter():
    def __init__(self, initial_state=np.array([0., 0., 0., 0]),
                 dt=1/20.,
                 measurement_uncertainty_x=2,
                 measurement_uncertainty_y=2,
                 process_uncertainty=1,
                 verbose=False, frame=1):
        self.name = "KalmanFilter"
        self.verbose = verbose
        self.kf = KF(dim_x=4, dim_z=2)

        self.kf.x = initial_state    # state (x and dx)
        self.process_uncertainty = process_uncertainty
        self.measurement_uncertainty_x = measurement_uncertainty_x
        self.measurement_uncertainty_y = measurement_uncertainty_y
        self.frame = frame
        self.frames = []
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])   # state transition matrix

        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])    # Measurement function

        self.kf.R = np.array([[self.measurement_uncertainty_x**2, 0],
                              [0, self.measurement_uncertainty_y**2]])
        self.Q = np.array([
            [(dt**4)/4, 0, (dt**3)/2, 0],
            [0, (dt**4)/4, 0, (dt**3)/2],
            [(dt**3)/2, 0, dt**2, 0],
            [0, (dt**3)/2, 0, dt**2]
        ])
        self.kf.Q = self.Q * self.process_uncertainty
        if self.verbose:
            print("Initializing Kalman Filter:\ndf = {dt}".format(dt=dt))

    def update_process_uncertainty(self):
        self.kf.Q = self.Q * self.process_uncertainty

    def update_measurement_uncertainty(self):
        self.kf.R = np.array([[self.measurement_uncertainty_x**2, 0],
                              [0, self.measurement_uncertainty_y**2]])

    def increase_process_uncertainty(self, alpha=0.01):
        self.process_uncertainty *= (1 + alpha)

    def decrease_process_uncertainty(self, alpha=0.01):
        self.process_uncertainty *= (1 - alpha)

    def smooth(self, zs, decrease_process_uncertainty=False, occluded=[]):

        smoothed_trajectory = []
        predicted_trajectory = []
        for i, z in enumerate(zs):
            if len(occluded) > 0:

                if occluded[i]:

                    self.measurement_uncertainty_x = 0.1
                    self.measurement_uncertainty_y = 0.1
                    self.process_uncertainty = 0.1

                else:
                    self.measurement_uncertainty_x = 0.1
                    self.measurement_uncertainty_y = 0.1
                    self.process_uncertainty = 0.1
                self.update_measurement_uncertainty()
            self.kf.predict()
            predicted_trajectory.append(
                np.array([[self.kf.x[0], self.kf.x[1]]]))

            if not np.isnan(z[0]):

                self.kf.update(z)
            smoothed_trajectory.append(
                np.array([[self.kf.x[0], self.kf.x[1]]]))

            if decrease_process_uncertainty:
                self.decrease_process_uncertainty()
                self.update_process_uncertainty()

        return np.concatenate(smoothed_trajectory), np.concatenate(predicted_trajectory)

    def step(self, z, decrease_process_uncertainty=False, frame=None):
        if frame > self.frame:
            self.kf.predict()
            pred_x = self.kf.x
        assert frame not in self.frames
        self.update(z, frame=frame)
        if decrease_process_uncertainty:
            self.decrease_process_uncertainty()
            self.update_process_uncertainty()

    def predict(self, frame=None,   increase_process_uncertainty=False):
        if increase_process_uncertainty:
            self.increase_process_uncertainty()
            self.update_process_uncertainty()
        self.kf.predict()
        self.frame = frame
        x = np.expand_dims(self.kf.x[:2], axis=0)

        return x

    def update(self, x, frame=None):
        self.kf.update(x)
        if frame is not None:
            self.frames.append(frame)

    def predictSequence(self, time):
        output = []
        for t in time:
            output.append(self.predict())
        return np.concatenate(output)
