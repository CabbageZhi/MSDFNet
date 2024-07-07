# Pytorch code for MSDFNet
MSDFNet: Multi-Scale Detail Feature Fusion Encoder-Decoder Network for Self-Supervised Monocular Thermal Image Depth Estimation    
![image](https://github.com/CabbageZhi/MSDFNet/assets/103650178/4b83ad91-f348-411f-ae22-bc225b457d47)

## Requirement
This code was developed and tested with python 3.7, Pytorch 1.5.1, and CUDA 10.2 on Ubuntu 16.04.
## Dataset
For ViViD Raw dataset, download the dataset provided on the [official website](https://sites.google.com/view/dgbicra2019-vivid/).  
For post-processed ViViD++ dataset, please download the dataset provided on the [link](https://urserver.kaist.ac.kr/publicdata/ViViD++/download_links.txt).  
After download our post-processed dataset, unzip the files to form the below structure.  
#### Expected dataset structure for the post-processed ViViD dataset:
```
KAIST_VIVID/
  calibration/
    cali_ther_to_rgb.yaml, ...
  indoor_aggressive_local/
    RGB/
      data/
        000001.png, 000002.png, ...
      timestamps.txt
    Thermal/
      data/
      timestamps.txt
    Lidar/
      data/
      timestamps.txt
    Warped_Depth/
      data/
      timestamps.txt
    avg_velocity_thermal.txt
    poses_thermal.txt
    ...
  indoor_aggressive_global/
    ...	
  outdoor_robust_day1/
    ...
  outdoor_robust_night1/
    ...
```

Upon the above dataset structure, you can generate training/testing dataset by running the script.
```bash
sh scripts/prepare_vivid_data.sh
```
## Train

```bash
sh scripts/trai_indoor.sh
sh scripts/train_outdoor.sh
```

## Evaluation

```bash
bash scripts/test_indoor.sh
bash scripts/test_outdoor.sh
```

