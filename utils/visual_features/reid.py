import tqdm
import os
import os.path as osp
import itertools

from torchvision.transforms import ToTensor
from torchvision import transforms
import PIL.Image as Image
import copy
import torch
import torch.nn.functional as F
from torchreid.data.transforms import build_transforms
from torchreid import models
import pandas as pd
import sklearn
import argparse
from tqdm import tqdm
import numpy as np
import traceback

import warnings
warnings.filterwarnings('ignore')

to_tensor = ToTensor()
to_pil = transforms.ToPILImage()


def pad_bbs(img, im, row_unclipped, padding='zero'):
    """
    im: padded image
    area_out: the area of the bb that is outside of image
    """

    left_pad = abs(int(row_unclipped['bb_left'])) if int(
        row_unclipped['bb_left']) < 0 else 0
    right_pad = abs(
        int(
            row_unclipped['bb_right']) -
        img.shape[2]) if int(
        row_unclipped['bb_right']) > img.shape[2] else 0
    top_pad = abs(int(row_unclipped['bb_top'])) if int(
        row_unclipped['bb_top']) < 0 else 0
    bot_pad = abs(int(row_unclipped['bb_bot']) - img.shape[1]
                  ) if int(row_unclipped['bb_bot']) > img.shape[1] else 0

    h = (row_unclipped['bb_bot'] - row_unclipped['bb_top'])
    w = (row_unclipped['bb_right'] - row_unclipped['bb_left'])
    area_out = (left_pad + right_pad) * h + (top_pad + bot_pad) * w
    area_out = area_out / (h * w)

    # zero padding
    if padding == 'zero':
        m = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bot_pad))
        im = m(im)

    # padding with image mean
    elif padding == 'mean':
        im = F.pad(
            im,
            (left_pad,
                right_pad,
                top_pad,
                bot_pad),
            "constant",
            im.mean())

    # padding with channel wise mean
    elif padding == 'channel_wise_mean':
        im = torch.stack([F.pad(im[i, :, :], (left_pad, right_pad, top_pad, bot_pad), "constant", im.mean(
            dim=1).mean(dim=1)[i]) for i in range(im.shape[0])])

    # 'circular'
    else:
        left_pad = left_pad if left_pad < im.shape[2] else im.shape[2] - 1
        right_pad = right_pad if right_pad < im.shape[2] else im.shape[2] - 1
        top_pad = top_pad if top_pad < im.shape[1] else im.shape[1] - 1
        bot_pad = bot_pad if bot_pad < im.shape[1] else im.shape[1] - 1
        im = F.pad(
            im.unsqueeze(0),
            (left_pad,
                right_pad,
                top_pad,
                bot_pad),
            padding).squeeze()

    return im, area_out


def get_transform():
    _, trans = build_transforms(
        height=256,
        width=129,
        transforms='random_flip',
    )
    return trans


def _get_images(frame_dets, trans, pad=False, dev='gpu'):
    res, area_out = list(), list()
    # get and image
    img = to_tensor(Image.open(
        frame_dets['frame_path'].unique()[0]).convert("RGB"))
    frame_size = (img.shape[1], img.shape[2])

    frame_dets['bb_right'] = frame_dets['bb_left'] + frame_dets['bb_width']
    frame_dets['bb_bot'] = frame_dets['bb_top'] + frame_dets['bb_height']

    frame_dets_cli = copy.deepcopy(frame_dets)
    frame_dets_cli['bb_right'] = frame_dets_cli['bb_right'].clip(
        0, frame_size[1])
    frame_dets_cli['bb_left'] = frame_dets_cli['bb_left'].clip(
        0, frame_size[1])
    frame_dets_cli['bb_top'] = frame_dets_cli['bb_top'].clip(0, frame_size[0])
    frame_dets_cli['bb_bot'] = frame_dets_cli['bb_bot'].clip(0, frame_size[0])

    frame_dets['outside'] = ((frame_dets_cli['bb_right'] == frame_dets['bb_right']) &
                             (frame_dets_cli['bb_left'] == frame_dets['bb_left']) &
                             (frame_dets_cli['bb_top'] == frame_dets['bb_top']) &
                             (frame_dets_cli['bb_bot'] == frame_dets['bb_bot'])) | \
        ((frame_dets['bb_left'] == frame_dets['bb_right']) |
         (frame_dets['bb_bot'] == frame_dets["bb_top"]))

    # iterate over bbs in frame
    for ind, row in frame_dets.iterrows():
        # get unclipped detections and bb (im)

        row_cli = frame_dets_cli.loc[ind]
        im = img[:, int(row_cli['bb_top']):int(row_cli['bb_bot']), int(
            row_cli['bb_left']):int(row_cli['bb_right'])]
        if not row.outside:
            res.append(torch.zeros([3, 256, 129]))
            continue
        # pad if part of bb outside of image
        if pad:
            im, area_out = pad_bbs(img, im, row)
        # transform bb
      
        im = to_pil(im)
       
        im = trans(im)
        # append to bbs, detections, tracktor ids, ids and visibility
        res.append(im)
    res = torch.stack(res, 0)
    res = res.to(dev)

    return res, frame_dets['outside'].values


def main(
        path_to_data,
        subset='all',
        det_file_name='det.txt',
        feature_folder="",
        pad=False,
        weight_path=None,
        sequence="",
        dev='cpu'):
    """
    subset: ['all', 'train', 'test']
    """
    trans = get_transform()
    model = models.build_model(name='resnet50', num_classes=1000).to(dev)
    if weight_path is not None:
        chkpt = torch.load(weight_path, map_location=torch.device(
            'cpu'), encoding='latin1')
        model.state_dict = chkpt['state_dict']

    model.eval()

    if subset == 'all':
        directories = ['train', 'test']
    else:
        directories = [subset]
    dets = pd.read_csv(
        det_file_name,
        names=[
            'frame',
            'id',
            'bb_left',
            'bb_top',
            'bb_width',
            'bb_height',
            'conf',
            'label',
            'vis',
            '?'])

    def add_frame_path(i):
        if type(i) == float:
            i = int(i)
        return osp.join(osp.join(path_to_data, f"{i:06d}.jpg"))

    dets['frame_path'] = dets['frame'].apply(add_frame_path)
    dets.sort_values("frame", inplace=True)
    first_frames = dets[dets.groupby(["id", "frame"]).frame.transform(
        max) == dets['frame']]

    dets["diff_t"] = dets.sort_values(
        ['id', 'frame']).groupby('id')['frame'].diff()
    dets["diff_t_back"] = dets.sort_values(
        ['id', 'frame']).groupby('id')['frame'].diff(-1)
    dets["diff_t"].fillna(-1, inplace=True)
    dets["diff_t_back"].fillna(-1, inplace=True)
    dets["step"] = (dets["diff_t"] == 1)*1
    dets["step"] += (dets["diff_t_back"] == -1)*1

    dets_features = dets[dets.step >= 1]
    

    outside_masks = list()
    feature_list = list()
    
    id_list = list()
    frame_list = list()
    for frame in tqdm(dets_features['frame'].unique()):

        df = dets_features[dets_features['frame'] == frame]
        ids = df.id.values
        try:
            bb_imgs, outside_mask = _get_images(df, trans, pad, dev)
            bb_imgs = bb_imgs[outside_mask]
            ids = ids[outside_mask]
            if len(bb_imgs) == 0:
                continue

            with torch.no_grad():
                features = model(bb_imgs)
        except:
            print(traceback.print_exc())

        feature_list.extend(features.cpu().numpy())
        outside_masks.extend(outside_mask.tolist())
        id_list.extend(ids.tolist())
        frame_list.extend([frame] * len(ids))

    final_array = np.concatenate((np.array(frame_list)[:, np.newaxis], np.array(
        id_list)[:, np.newaxis], np.array(feature_list)), 1)

    # store features
    os.makedirs(feature_folder, exist_ok=True)
    np.save(os.path.join(feature_folder, f"{sequence}.npy"), final_array)
   

def compute_dist(X, Y=None, distance_metric='cosine'):
    if Y is None:
        Y = X
    dist = sklearn.metrics.pairwise_distances(
        X,
        Y=Y,
        metric=distance_metric)

    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic experiment setting
    parser.add_argument('--challenge', default="MOT17",
                        type=str, help="challenge")
    parser.add_argument('--sequences', nargs="+", default=["MOT17-02"],
                        type=str, help="sequence")
    parser.add_argument('--trackers', nargs="+", default=["CenterTrack"],
                        type=str, help="tracker")
    args = parser.parse_args()

    for sequence, tracker in itertools.product(args.sequences, args.trackers):
        print(f'Running sequence: {sequence} and tracker: {tracker}')
        feature_folder = f"./data/{args.challenge}/tracker/{tracker}/features"
       
        main(

            path_to_data=f"./data/{args.challenge}/sequences/{sequence}/img1/",
            subset='all',
            sequence=sequence,
            det_file_name=f"./data/{args.challenge}/tracker/{tracker}/data/{sequence}.txt",
            feature_folder=feature_folder,
            weight_path='./data/reID_weights/resnet50_market_xent.pth.tar',
            pad=False,
            dev='gpu' if torch.cuda.is_available() else 'cpu')
