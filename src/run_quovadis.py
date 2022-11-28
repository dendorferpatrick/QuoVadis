import argparse
import os


from quovadis.datasets import MOTData


from quovadis.utils import prepare_sequence, get_cfg
from quovadis.tracker import MOTTracker

from quovadis import Evaluator, QuoVadis
from quovadis.trajectory_predictor import get_predictor


def get_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for running Quo Vadis Tracker")
    parser.add_argument(
        "--config-file",
        default="run/cfgs/base.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="MOT17",
        help="List of sequences to run",
    )

    parser.add_argument(
        "--tracker-name",
        type=str,
        default="CenterTrack",
        choices=['ByteTrack', 'CenterTrack', 'qdtrack',
                 'CSTrack', 'FairMOT', 'JDE', 'TraDeS', 'TransTrack'],
        help="Name of underlying tracker",
    )

    parser.add_argument(
        "--sequences",
        nargs='+',
        default=["MOT17-02"],
        help="List of sequences to run",
    )

    parser.add_argument(
        "--frames-start-end",
        nargs='+',
        default=[],
        help="Enter start and end variable",
    )

    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save result file",
    )

    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="Run evaluation after running the tracker",
    )

    parser.add_argument(
        "--vis-results",
        default=False,
        action="store_true",
        help="Plot results",
    )

    parser.add_argument(
        "--save-vis",
        default=False,
        action="store_true",
        help="Save visualization",
    )

    parser.add_argument(
        "--make-video",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()

    # load cfg
    cfg = get_cfg(args)

    # load dataset and sequences
    mot = MOTData(sequences=cfg.sequences,
                  dataset=cfg.dataset,
                  fields=["rgb",
                          "tracker",
                          "map",
                          "homography",
                          "egomotion"
                          ])

    predictor = get_predictor(cfg)
    for sequence in mot.data.sequences:

        tracker_df, y0 = prepare_sequence(sequence)

        tracker = MOTTracker(df=tracker_df,
                             tracker_name=cfg.tracker_name,
                             smooth_positions=cfg.SMOOTH_POSITIONS,
                             visual_features=os.path.join(cfg.DATA_DIRECTORY,
                                                          cfg.dataset,
                                                          "tracker",
                                                          cfg.tracker_name,
                                                          "features",
                                                          f'{sequence.name}.npy'))

        # set frames to run Quo Vadis on
        if len(cfg.frames_start_end) == 2:
            start_frame = max(tracker.df.frame.min(), cfg.frames_start)
            end_frame = min(tracker.df.frame.max(), cfg.frames_end) + 1
        else:
            start_frame = tracker.df.frame.min()
            end_frame = tracker.df.frame.max() + 1
        frames = list(range(start_frame, end_frame))

        quo_vadis = QuoVadis(predictor=predictor,
                             tracker=tracker,
                             sequence=sequence)
        # run model
        quo_vadis.run(motion_dim=cfg.MOTION_DIM,
                      save_results=cfg.save,
                      save_directory=cfg.SAVE_DIRECTORY,
                      min_iou_threshold=cfg.MIN_IOU_THRESHOLD,
                      L2_threshold=cfg.L2_THRESHOLD,
                      IOU_threshold=cfg.IOU_THRESHOLD,
                      min_appearance_threshold=cfg.APP_THRESHOLD,
                      frames=frames,
                      reId_metric=cfg.REID_METRIC,
                      visibility=True,
                      max_age=cfg.MAX_AGE,
                      max_age_visible=cfg.MAX_AGE_VIS,
                      debug=True,
                      y0=y0)

        if cfg.make_video or cfg.vis_results:

            quo_vadis.vis_results(show=False,
                                  save=cfg.save_vis,
                                  make_video=cfg.make_video,
                                  frames=frames)

    if args.eval:
        evaluator = Evaluator()
        for sequence in mot.data:
            evaluator.eval(args.dataset, [sequence.name], [
                           args.tracker_name], quo_vadis.tracker_folder)
        evaluator.eval(args.dataset, 
                        mot.data.sequence_names, [
                       args.tracker_name], quo_vadis.tracker_folder)
        evaluator.print_results()


if __name__ == "__main__":
    main()
