import shutil
import random
import os 
import numpy as np
from PIL import Image
train  = 0.65
val = .15 
test = .2

root_dir = "/storage/user/dendorfp/MOTSynth/dataset"

all_sequences = os.listdir(os.path.join(root_dir, "xy"))
random.shuffle(all_sequences)
n_train = int(len(all_sequences) * train)
n_val =  n_train  + int(len(all_sequences) * val)
n_test =  n_val  + int(len(all_sequences) * test)

print(n_train, n_val, n_test, len(all_sequences))


train_files = all_sequences[:n_train]
val_files = all_sequences[n_train:n_val]
test_files = all_sequences[n_val:]

print(train_files, val_files, test_files)
data = {"train": train_files, "val": val_files, "test": test_files}

for key , values in data.items():
    for k in ["xy", "uv"]:
        directory = os.path.join(root_dir, "motsynth_{}".format(k), key)
        try:
            os.makedirs(directory)
        except: 
            pass
        for file in values: 
            print(key, file)
            shutil.copyfile(os.path.join(root_dir, k, file),
            os.path.join(directory, "{}_{}".format(key, file)) )


            array = np.ones((1080,1920, 3),  np.uint8) * 255
           

            im = Image.fromarray(array)
            im.save(os.path.join(directory, "{}.jpg".format(file.split(".")[0])))