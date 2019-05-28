# Change-Background
Main characters (persons) extraction into new (desired) background utilizing the Mask R-CNN

## Introduction
This project mainly aims to change the background (Just need an image of your desired background) of input file (Image or video) 
while keeps the main charactors (person) same as input file.
This project uses [MASK R-CNN](https://github.com/matterport/Mask_RCNN) for object detection.
## Demo

| Demo 1        | Demo 2           |
| ------------- |:-------------:|
| <img src = 'demo_results/video_result_1.gif'>     | <img src = 'demo_results/video_result_2.gif'> |

## Requirements
numpy

scipy

cython

h5py

Pillow

scikit-image

tensorflow-gpu>=1.3 (Tested on 1.13)

keras

jupyter

matplotlib

imgaug (It requires shapely to be installed)

IPython[all]

opencv-python

## Getting started
Ready the environment by installing all the packges. If you are facing any issues in setting up the environment then you can refer to 
Mask RCNN page. This project doesn't need any extra packages to run.
Before running, please download mask_rcnn_coco.h5 file [by clicking here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) and put it into the main directory

Now, you can run the "main_single_person_extraction.ipynb" and follow the Instructions to process video and image files separately.

## Upcoming
A general program will be added in which you can specify No of persons you want to extract from the input file. 
(Right now It extract only a single person who's area is biggest)

## Ref
Mask RCNN
