# Body-part Aware Network and Object Affordance Masking mechanism for Human-Object-Interaction Detection (BANAM)

## Introduction

This is official implementation of BANAM methods for Human-Object-Interaction Prediction developed by Nguyen Trong Tung and Nguyen Ho Huu Nghia for our thesis graduation project.
In this work, we presented a novel method following two-stage approaches which can utilize different human-centric information for making prediction on different relationship categories
between human and object. We experimented our methods on [HOI-A test 2019 dataset](http://www.picdataset.com/challenge/index/) and achieved state-of-art results comparing to other 
two-stage approach methods

## Main Results
### Results on HOIA-test 2019
| Arch               | mAP@0.5 |
|--------------------|------|
| iCAN     | 44.23 | 
| TIN    | 48.64 | 
| GMVM    | 60.26 | 
| C-HOI | 66.04 |   
| Ours BANAM | 66.17 |  

## Methods
### Overview
We proposed to decompose HOI problem into two main stages: object detection and HOI prediction. For the second stage, we broke into two sub-stage: pair matching and relationship prediction.
This was motivated by our observations that pair matching stage can help to eliminate irrelevant pairs before giving prediction of relationship categories for each pair
![General Pipeline](/figures/hoi_decomposition.png)

### Object Detection
Since this is not our main contribution, we delegated this component to an external object detector and experimented with different model architectures for choosing the appropriate solution for our system.

### Pair Matching

We proposed a pair matching network serving as an approximated pair-matching function for predicting matching scores for a given pair. In this part, we considered two crucial features for making prediction on human-pair matching which are global features and body-part aware features. The general pipeline for pair matching network and architectures for constructing global and body-part aware features are shown below

![Pair matching pipeline](/figures/pair_matching.png)
![Global features](/figures/glob_feat.png)
![Body_part_aware_features](/figures/bpa.png)

### Relationship Prediction

Our final components of the system is relationship prediction. In this part, we introduced a novel type of feature call object affordance feature. This type of features were used as prior knowledge about the semantic association between an object and relationship. For example, it is nonsense to associate relationship "eating" to object categories "bicycle". We made use of this feature in post processing step for masking our irrelevant relationship categories scores. The overall pipeline of our final component was shown below
![Rel_prediction](/figures/masking_affordance.png)

## Repository Usage
#### Installation

Run the following commands for setting up the project

```
git clone https://github.com/nttung1110/Thesis-Draft

conda create --name BANAM --file requirement.txt

conda activate BANAM
```

#### Data

The HOI-A 2019 dataset was released at PIC challenge in ICCV 2019 workshop. To download the data, you are suggested to register for
an account and go to [HOI-A 2019](http://picdataset.com/challenge/index/) for downloading it

### Training

The second-stage training process is divided into two sub-stage: pair matching network and relationship prediction network which are trained independently.

The input to both network similarly requires object detection results from the previous stage. As we have mentioned before,
the object detection module can be adaptively changed depending on the use cases. Therefore, the users can update object detection model
with many state-of-the-art architecture such as: YOLO, Efficient-Det, ... However, the output of object detection model have to follow the
following format for supplying to our second-stage. Output must be in json format which is similar to the following examples:

```
{
  {
    "file_name": "trainval_000000.png",
    "annotations": [
      {
          "bbox": [
              273,
              93,
              458,
              480
          ],
          "category_id": "1"
      },
      {
          "bbox": [
              354,
              149,
              364,
              156
          ],
          "category_id": "3"
      }
    ]
  },
  {
    "file_name": "trainval_000000.png",
    ...
  },
  
}
```

* `file_name`: name of image.
* `annotations`: list of bounding boxes being predicted by object detection model, each element stores bounding box along with object category of the result.

Note that object detection results json file should be put in ```./detect_result``` folder
#### Pair Matching Network

In this sub-stage, we utilized pose estimation results as additional information for training our network. Therefore, we employed an off-the-self 
keypoint estimation model from [Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
. The users are suggested to extract keypoint results by using the pre-installed folder which we have set up.

For preparing the keypoint of human existing inside the image, follow the following commands:

```
cd external_lib
python infer_pose_2_json.py
```
There are some parameters which should be changed inside ```infer_pose_2_json.py``` script for external usage
* `path_img_in_test`: path to image folder to be trained
* `path_json_detect_test`: path to detection result which have been predicted by object detector. For example ```../detect_result/YOLO_result.json````
* `path_json_out_pose_test`: path to keypoint estimation result in json format

After human keypoint result of the image folder have been extracted, the result will be stored in json file which should be transferred into 
folder ```pose-data-2019``` for later usage

Finally, the pair matching network are ready to be trained. Some of the parameters should be specified before running the script:

```
cd script
mkdir checkpoint
python train_pair_matching --lr 0.001 --batch_size 64 --weight-decay 1e-4 --num_workers 8 --epochs 100 --optimizer sgd
```

* `data_root`: path to root of the HOI-2019 dataset
* `data_root_2`: path to cache data for storing feature extracted by feature extractor
* `data_root_pose`: path to keypoint estimation result in json format

Weight for epochs will be stored in folder ```./checkpoint/pair_matching/```

#### Relationship Prediction Network

This network was trained in a much simpler way than pair matching network. 

### Visualization

### Citation

### Acknowledgement

### Contact Information

If you have any concerns about this project, please contact:

+ Nguyen Trong Tung(ntrtung17@apcs.vn)

+ Nguyen Ho Huu Nghia(nhhnghia@apcs.vn)
