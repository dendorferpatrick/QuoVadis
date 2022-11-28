# Bird's-Eye View Reconstruction

To run MG-GAN as our trajectory predictor in our tracking model, we need to estimate the homography transformation between the ground plane and the image. Here, we assume that the objects move on a flat ground plane. As shown in the figure, depth estimates, panoptic segmentation, 
Before you start make sure that you have everything installed following [docs/INSTALL.md](docs/INSTALL.md). Furthermore, you need to place the camera intrinsics matrix (3x3 matrix, comma separated) for the sequences you want to run into ``./data/<dataset_name>/sequences/<seq>/calib/calib.txt``.

<div align="center">
    <img src="/misc/bev_reconstruction.png" width="80%" alt=""/>
</div>


1. Run Depth estimator  
    Make sure you have downloaded the pretrained model ["AdaBins_synthetic.pt"](https://drive.google.com/file/d/1HMQJI01n3ncH8mOxb3-F3uQX0fNsg83h/view?usp=sharing) and placed it into the path: ``./data/AdaBins_weights/AdaBins_synthetic.py``. 
   
    Run the depth model on the first image of the sequence 

    ```
    python src/run_depth_estimator.py --model_path <checkpoint_path_pt> --input_path ./data/<dataset>/sequences/<sequence_name>/img1 --output_path ./data/<dataset>/sequences/<sequence_name>/depth --images 000001.jpg
    ```

2. Run Panoptic Segmentation

    ```
    python src/run_panoptic_segmentation --dataset <dataset_name> --sequences <seq1> <seq2> ...
    ```
    You can add the argument `--single` if you only want to run segmentation on the first frame.
3. Estimate Homography

    For static sequences, it is sufficient to have depth estimates and panoptic segmentation for a single image.
    ```
    python run/run_homography.py --dataset <dataset_name> --sequnece <sequence_name>
    ```
