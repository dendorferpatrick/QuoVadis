import json
import os
DATAPATH = "/storage/user/dendorfp/MOTSynth/frames"


def compute_splits(split = "train"):
    train = 0.4
    val = 0.2
    assert (train + val) < 1
    test = 1- val - train

    seqs = os.listdir(DATAPATH)
    exclude = ["629", "757"]
    for s in exclude:

        seqs.remove(s)

    print(len(seqs))

    import random
    random.Random(4).shuffle(seqs)
    print(seqs)
    
    train_index_start = 0 
    train_index_end = val_index_start = int(len(seqs) * train)
    val_index_end =  test_index_start = int(len(seqs) * (train + val))
    test_index_end = len(seqs)

    train_data = seqs[train_index_start: train_index_end]
    val_data = seqs[val_index_start: val_index_end]
    test_data = seqs[test_index_start: test_index_end]

    for s in ["327", "455", "524", "652"]:
        if s in train_data: train_data.remove(s)
        if s in val_data: val_data.remove(s)
        if s in test_data: test_data.remove(s)


    print(len(train_data) + len(val_data) + len(test_data))

    if split == "train": 
        return train_data
    elif split == "val":
        return val_data
    elif split == "test":
        return test_data
    elif split == "all":
        return train_data, val_data, test_data
    else: 
        return False


if __name__ == "__main__":
    # train_data, val_data, test_data = compute_splits("all")
    # print(len( train_data), len(val_data) , len(test_data))
    outfolder = "/usr/wiss/dendorfp/dvl/projects/TrackingMOT/splits/MOTSynth"
    os.makedirs(outfolder, exist_ok= True)
    data_dir = "/storage/user/dendorfp/MOT16/MOTSynth_depth" 
    sequences = os.listdir(data_dir)
    
    
    # ready_sequences = os.listdir("/storage/user/dendorfp/MOTSynth/positions_depth/")
    # ready_sequences = [seq.split(".")[0] for seq in ready_sequences]
   
    # sequences = set(sequences).difference(set(ready_sequences))
    
    # for sequence in np.arange(1):
    #     # args = parser.parse_args()
    #     create_position(sequence)

    exclude_sequences = [629, 757, 524, 652]
    test_data =  [int(seq) for seq in sequences if int(seq) not in exclude_sequences] 
    print(test_data)
    with open(os.path.join(outfolder, "test.json"), 'w') as outfile:
        json.dump(test_data, outfile)