import os 
import itertools
import time


challenge = "MOT16"
tracker_list =  ['ByteTrack', 'CenterTrack', 'qdtrack', 'CSTrack', 'FairMOT', 'JDE', 'TraDeS', 'TransTrack'][:1]
if challenge == "MOT16": sequences = [8, 7, 3, 1]
elif challenge == "MOT20": sequences = [5]

for tracker_name, sequence in itertools.product(tracker_list, sequences):
    feature_folder = f"/storage/user/dendorfp/{challenge}/tracker/{tracker_name}/features"
    
    if os.path.exists(os.path.join(feature_folder, f"{challenge}-{sequence:02d}.npy")): 
        print("Features already exist")
        continue

    command = f"sbatch run_slurm.sbatch reid_for_patrick.py --tracker {tracker_name} --sequence {challenge}-{sequence:02d} --challenge {challenge}"
    os.system(command)
  
