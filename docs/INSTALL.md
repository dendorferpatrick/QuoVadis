# Installation
1. Clone the repository: 
    ```
    git clone --recurse-submodules https://github.com/dendorferpatrick/QuoVadis
    cd QuoVadis
    ```

2. Install packages with Python >= 3.6.
    1. `pip install -r requirements`
    2. `pip install -e .` 
    3. `pip install -e submodules/MGGAN/`  
    4. `pip install -e submodules/detctron2/` (optional, only needed for panoptic segmentation)  
    5. `pip install -r submodules/AdaBins/requirements.txt` (optional, only needed for depth estimation)  
    6. `pip install -r submodules/AdaBins/` (optional, only needed for depth estimation) 
    7. `pip install -r submodules/deep-person-reid/` (optional, only needed for feature extraction)


    Alternatively, you can run `./setup_all.sh`. This script starts all the installations shown above.

    If you want to run Quo Vadis on MOT17 and/or MOT20 and want to use our pre-trained and preprocessed data, also follow the next steps.

3. Download datasets

    In this repository, we run experiments on the ([MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/))          datasets.
    To download the sequences you can either run the download files in `./download` using `wget` or download the data yourself.

    For MOT17:
    ````
    bash downloads/MOT17_data.sh
    ````

    For MOT20: 
    ```
    bash downloads/MOT20_data.sh
    ````

4. Download pre-processed data
    Quo Vadis requires several input files, data, and the pretrained motion model weights.
    
    - Homographies of the sequences
    - Tracker output and visual features
    - Map Information of the scene
    - Pretrained Motion Model

    To download the required data, please run the following scripts. The data download will take some minutes.

    For MOT17:
    ```
    bash downloads/download_MOT17_all.sh
    ```
    For MOT20:
    ```
    bash downloads/download_MOT20_all.sh
    ```

After a successful download the files structure in the ```./data``` should look the following:

~~~
    |-- data
        |-- predictor_weights
        |   |-- <predictor_model>
        |   |   |-- checkpoints
        |   |   |   |-- <checkpoint_name>.pth
        |   |   |-- meta_tags.csv
        |-- <dataset_name>
        |   |-- sequences
        |   |   |-- <seq1>
        |   |   |-- <seq2>
        |   |   |   |-- egomotion
        |   |   |   |-- gt
        |   |   |   |   |-- gt.txt
        |   |   |   |-- homography
        |   |   |   |   |-- homography.json
        |   |   |   |-- img1
        |   |   |   |   |-- 000001.jpg
        |   |   |   |   |-- 000002.jpg
        |   |   |   |   |-- ...
        |   |   |   |-- map
        |   |   |   |   |-- rgb.png
        |   |   |   |   |-- visibility.png
        |   |-- tracker
        |   |   |-- <tracker_name>
        |   |   |   |-- data
        |   |   |   |   |-- <seq1>.txt
        |   |   |   |   |-- <seq2>.txt
        |   |   |   |   |-- ...
        |   |   |   |-- features
        |   |   |   |   |-- <seq1>.npy
        |   |   |   |   |-- <seq2>.npy       
        
~~~
