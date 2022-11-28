import os
from collections import namedtuple

from quovadis.datasets import utils
from quovadis.datasets.dataset import Sequence, DataSet


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
            "depth": Field("depth", "depth", "sequences", utils._depth_setup_cb, utils._depth_frame_cb),
            "calibration": Field("calibration", "calib", "sequences", utils._calibration_setup_cb, utils._calibration_frame_cb),
            "labels": Field("labels", "labels", "sequences", utils._labels_setup_cb_mot, utils._labels_frame_cb),
            "rgb": Field("rgb", "img1", "sequences", utils._rgb_setup_cb, utils._rgb_frame_cb),
            "dets": Field("dets", "det", "sequences", utils._dets_setup_cb, utils._dets_frame_cb),
            "panoptic": Field("panoptic", "panoptic", "sequences", utils._panoptic_setup_cb, utils._panoptic_frame_cb),
            "segmentation": Field("segmentation", "segmentation", "sequences", utils._segmentation_setup_cb, utils._segmentation_frame_cb),
            "map": Field("map", "map", "sequences", utils._map_setup_cb, utils._map_frame_cb),
            "tracker": Field("tracker", "data", "tracker", utils._tracker_setup_cb, utils._tracker_frame_cb),
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

    mot = MOTData(sequences=["MOT17-02"], dataset="MOT17",
                  prefix="./data", fields=["rgb", "panoptic"])
    mot.data.sequences[0].plot_panoptic(1)