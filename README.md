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
### Environment Settings

#### Installation

#### Data

### Training and Testing

### Visualization

### Citation

### Acknowledgement

### Contact Information

If you have any concerns about this project, please contact:

+ Nguyen Trong Tung(ntrtung17@apcs.vn)

+ Nguyen Ho Huu Nghia(nhhnghia@apcs.vn)
