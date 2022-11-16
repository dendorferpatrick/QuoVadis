import torch
import numpy as np

from data_utils.TrajectoryDataset import (
    TrajectoryDataset,
    seq_collate,
)


def get_dataloader(
    dataset,
    phase,
    augment=False,
    batch_size=8,
    workers=0,
    shuffle=False,
    split=None,
    load_semantic_map = False, 
    grid_size = 16, 
    pred_len = 12, 
    obs_len = 8 
):
    assert phase in ("train", "val", "test")

    if phase in ("val", "test") and augment is True:
        print("No augmentation during validation or testing.")
        augment = False

    if dataset in (
        "stanford_synthetic",
        "stanford_synthetic_2",
        "social_stanford_synthetic",
    ):
        ds = TrajectoryDataset(
            dataset_name=dataset,
            phase=phase,
            margin_in=16,
            margin_out=16,
            load_occupancy=False,
            scaling_small=1.2,
            data_augmentation=int(augment),
        )

        if split in ("upper", "lower"):
            if split == "lower":
                selector = ds.trajectory[:, 8, 1] > 16.0
            else:
                selector = ds.trajectory[:, 8, 1] <= 16.0

            new_scene_list = []
            new_trajectory = []
            new_ped_ids = []
            new_seq_start_end = []
            last_end = 0
            for scene_idx, (start, end) in enumerate(ds.seq_start_end):
                if selector[start:end].any():
                    new_scene_list.append(ds.scene_list[scene_idx])
                    new_trajectory.append(ds.trajectory[start:end])
                    new_ped_ids.append(ds.ped_ids[start:end])

                    next_end = last_end + end - start
                    new_seq_start_end.append([last_end, next_end])
                    last_end = next_end

            ds.trajectory = np.concatenate(new_trajectory)
            ds.ped_ids = np.concatenate(new_ped_ids)
            ds.seq_start_end = new_seq_start_end
            ds.scene_list = new_scene_list

        collate_fn = seq_collate
    elif dataset == "stanford":
        ds = TrajectoryDataset(
            dataset_name="stanford",
            phase=phase,
            margin_in=16,
            margin_out=16,
            load_occupancy=False,
            scaling_small=0.7,
            data_augmentation=int(augment),
        )
        collate_fn = seq_collate
    elif dataset.lower() in ("eth", "hotel", "zara1", "zara2", "univ", "gofp", "motsynth", "motsynth_uv"):
        ds = TrajectoryDataset(
            dataset_name=dataset,
            phase=phase,
            grid_size_in_global=grid_size,
            grid_size_out_global=grid_size,
            scene_batching=True,           
            goal_gan = False, 
            cnn = (grid_size > 0), 
            local_features = False, 
            scaling_global=0.5,
            load_semantic_map = load_semantic_map,
            data_augmentation=int(augment),
            pred_len = pred_len, 
            obs_len = obs_len
        )
        collate_fn = seq_collate
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader
