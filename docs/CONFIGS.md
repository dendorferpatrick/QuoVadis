# Configurations



## Data structure configs:
`DATA_DIRECTORY`: Directory path of data folder ``./data``  
`SAVE_DIRECTORY`: Path where results should be saves

## Trajectory Prediction

### Trajaectroy prediction general configs:
`MOTION_DIM`: Dimension of trjaectory prediction. If `2` prediction in image space, if `3` prediction in BEV  
`DT`: Temporal steplenght of prediction step (0.4)  
`SMOOTH_POSITIONS`: False  
`PRED_MODEL`: Prediction model. Implemented options (`static`, `linear`, `kalman`, `gan`, `mggan`)  

### GAN configs:
`DEVICE`: "cpu" or "cuda"  
`MGGAN_WEIGHTS`: Path to mggan pretrained weights  
`NR_PREDICTIONS`: 3  

## Parameters for matching:
`REID_METRIC`: Parameter for matching metric

### Thresholds for mathcing score
`L2_THRESHOLD`: Maximal L2 distance considered hungarian matching  
`IOU_THRESHOLD`: Minimal IoU score considered for hungarian matching  

### Thresholds for matching:
`MIN_IOU_THRESHOLD`: Minimal IOU threshold  
`APP_THRESHOLD`: Appearance threshold  

`MAX_AGE`: Maximal age of inactive tracks in frame  
`MAX_AGE_VIS`: Maximal number of visible frames for inactive prediction  


