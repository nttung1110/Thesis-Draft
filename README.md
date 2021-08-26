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
We proposed to decompose HOI problem into two main stages: object detection and HOI prediction. For the second stage, we broke into two sub-stage: pair matching and relationship prediction.
This was motivated by our observations that pair matching stage can help to eliminate irrelevant pairs before giving prediction of relationship categories for each pair
![Illustrating the architecture of the proposed BANAM](/figures/hoi_decomposition.pdf)
