import argparse
import os


from datasets import MOTData


from quo_vadis.utils import prepare_sequence, get_cfg
from tracker import MOTTracker

from quo_vadis import QuoVadis
from trajectory_predictor import get_predictor


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
        "--tracker-name",
        type=str,
        default="CenterTrack",
        choices=['ByteTrack', 'CenterTrack', 'qdtrack',
                 'CSTrack', 'FairMOT', 'JDE', 'TraDeS', 'TransTrack'],
        help="Name of underlying tracker",
    )

    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="Run evaluation after running the tracker",
    )

    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save result file",
    )

    parser.add_argument(
        "--plot_results",
        default=False,
        action="store_true",
        help="Plot results",
    )

    parser.add_argument(
        "--save_vis",
        default=False,
        action="store_true",
        help="Save visualization",
    )

    parser.add_argument(
        "--make_video",
        default=False,
        action="store_true",
        help="Baseline tracker",
    )

    parser.add_argument(
        "--sequences",
        nargs='+',
        default=["MOT17-02"],
        help="List of sequences to run",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="MOT17",
        help="List of sequences to run",
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
    for sequence in mot.data.sequences:
        # get sequence for run
        sequence = mot.data.sequences[0]

        tracker_df, y0 = prepare_sequence(mot, sequence)

        tracker = MOTTracker(df=tracker_df,
                             tracker_name=cfg.tracker_name,
                             smooth_positions=cfg.SMOOTH_POSITIONS,
                             visual_features=os.path.join(cfg.DATA_DIRECTORY, cfg.dataset, "tracker", cfg.tracker_name, "features",
                                                          f'{sequence.name}.npy'))

        frames = range(tracker.df.frame.min(), tracker.df.frame.max() + 1)
        frames = range(tracker.df.frame.min(), tracker.df.frame.min() + 5)
        
        predictor = get_predictor(cfg)

        motion_model = QuoVadis(predictor=predictor,
                                tracker=tracker,
                                sequence=sequence)

        # run model
        motion_model.run(motion_dim=cfg.MOTION_DIM,
                         save_results=cfg.save_results,
                         save_directory=cfg.SAVE_DIRECTORY,
                         min_iou_threshold=cfg.MIN_IOU_THRESHOLD,
                         L2_threshold=cfg.L2_THRESHOLD,
                         IOU_threshold=cfg.IOU_THRESHOLD,
                         min_appearance_threshold=cfg.APP_THRESHOLD,
                         hallucinate=False,
                         frames=frames,
                         social_interactions=False,
                         Id_switch_metric="IOU",
                         reId_metric=cfg.REID_METRIC,
                         visibility=True,
                         max_age=cfg.MAX_AGE,
                         max_age_visible=cfg.MAX_AGE_VIS,
                         debug=True,
                         exists_ok=True,
                         y0=y0)
        if cfg.save and cfg.eval:
            motion_model.run_eval()

        if cfg.make_video or cfg.plot_results:
            motion_model.plot_results(show=False, 
                                        save=cfg.save_vis, 
                                        make_video=cfg.make_video, 
                                        save_folder="tmp",
                                        frames=frames)


if __name__ == "__main__":
    main()
