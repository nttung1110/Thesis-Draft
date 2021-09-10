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

Our final component of the system is relationship prediction. In this part, we introduced a novel type of feature called object affordance feature. This type of features were used as prior knowledge about the semantic association between an object and relationship. For example, it is nonsense to associate relationship "eating" to object categories "bicycle". We made use of this feature in post processing step for masking our irrelevant relationship categories scores. The overall pipeline of our final component was shown below
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
* `path_json_detect_test`: path to detection result which have been predicted by object detector. For example ```../detect_result/YOLO_result.json```
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
* `data_root_2`: path to cache data for storing feature extracted by feature extractor which will be used by **pair matching network**
* `data_root_pose`: path to keypoint estimation result in json format

Weight for epochs will be stored in folder ```./checkpoint/pair_matching/```

#### Relationship Prediction Network

This network was trained in a much simpler way than pair matching network. Some of the parameters should be specified before running the script:

```
cd script
python train_sim_net
```

* `data_root`: path to root of the HOI-2019 dataset
* `data_root_2`: path to cache data for storing feature extracted by feature extractor which will be used by **relation prediction network**
* `path_ckpt`: path to checkpoint of model


### Evaluation 
For reproducing the results on HOI-A test 2019 dataset, the users are suggested to download our [Pretrained Weights](https://drive.google.com/drive/folders/11JVfVKBnv_1wqlziVwmauUPetkw8TeEF?usp=sharing) which includes the following
weights for three models:

* Stage 1 (Object Detection): During our experiment, we reported that YOLOv5 detection results have the highest scores
among other detectors. Therefore, we released our pre-trained weight of YOLOv5 architecture for HOI-A 2019 dataset

* Stage 2 (Pair matching and Relationship Prediction): 17 and 30 training epochs for pair matching and relationship prediction respectively

After that, check and specify some essential parameters as following and run the script:

```
cd script
python infer_full_pipeline.py
```

* `path_ckpt_predict`: path to pretrained weights of pair matching model
* `path_ckpt_matching`: path to pretrained weights of relationship prediction network
* `path_json_detect`: path to detection results in json file
* `path_image_folder`: path to image folder of the dataset
* `path_json_pose`: path to keypoint estimation results for pair matching model
* `path_json_out`: path to json file which stores results for evaluation 
* `path_json_construct_co_matrix`: path to `train_2019.json` for constructing object affordance features

Then, specify the path to results of above script in `evaluation.py` at parameter `pred_json` for 
viewing the final results of our model with mAP scores for 11 relationship categories in HOI-A test 2019 dataset.

### Visualization

We also prepared demo implementation for both images and video. However, the video results were constructed by extracting results individually for each frame
which will not utilize the temporal information. For running the demo, the users are suggested to dowload our [Pretrained Weights](https://drive.google.com/drive/folders/11JVfVKBnv_1wqlziVwmauUPetkw8TeEF?usp=sharing). 

In our implementation, we used YOLOv5 model architecture for predicting object bounding boxes and categories in a given image. The results will be
then processed by our BANAM network to predict human-object-interaction categories. The user should specify image path for predicting results 
with parameter `path_image`

```
cd script
python demo_single_image.py
```

The users can also change the object detector results by creating their own modules and 
specifying the result at line 175 of script `demo_single_image.py`


### Citation

### Acknowledgement

### Contact Information

If you have any concerns about this project, please contact:

+ Nguyen Trong Tung(ntrtung17@apcs.vn)

+ Nguyen Ho Huu Nghia(nhhnghia@apcs.vn)
