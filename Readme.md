# Domain-specific semantics guided approach to video captioning
The code for the paper “Domain-Specific Semantics Guided Approach to Video Captioning”, WACV 2020.

## Platform for working

* Python 3.5
* Keras built with tensorflow backend [Version 2.31]
* Tensflow [Version 1.15.0]

### Getting started

* Resnet-152 and C3D Features are provided in the data folder
* For semantic features we have provided Ground truth from SCN paper. You can train your own MLP to extract features
* You can evaluate the model using MSCOCO pycocotools


## Citation
Please cite our WACV paper if it helps your research:

@INPROCEEDINGS{Domain_vid_capt_wacv_2020,  
author={M. {Hemalatha} and C. C. {Sekhar}},  
booktitle={2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},   
title={Domain-Specific Semantics Guided Approach to Video Captioning},   
year={2020},  
pages={1576-1585},
}
 
