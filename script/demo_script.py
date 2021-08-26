import argparse
import time
from pathlib import Path
import sys
import pdb
# sys.path.insert(0, "/home/nhhnghia/yolov5")

import cv2
import torch
import torch.backends.cudnn as cudnn


from yolov5.demo import parse_opt, initialize, detect


def example(imgs):
    opt = parse_opt("")
    opt.weights = "/home/nhhnghia/yolov5/runs/train/yolov5x6-hoia-detection-mixup10/weights/last.pt"
    opt.device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # initialize the model
    model, names = initialize(**vars(opt))
    opt.img_size = 896
    opt.model = model 
    opt.names = names
    opt.save_img = True
    opt.save_crop = True
    opt.save_dir = './'
    preds = []
    for img in imgs:
        opt.pred_img = img
        # make the prediction
        pred = detect(**vars(opt))
        preds.append(pred)
    # pdb.set_trace()
    print(preds)
    return preds

def detect_single_img(path_img):
    sample_img = cv2.imread(path_img)
    imgs = [sample_img]
    opt = parse_opt("")
    opt.weights = "/home/nhhnghia/yolov5/runs/train/yolov5x6-hoia-detection-mixup10/weights/last.pt"
    opt.device= "cuda:0" if torch.cuda.is_available() else "cpu"
    # initialize the model
    model, names = initialize(**vars(opt))
    opt.img_size = 896
    opt.model = model 
    opt.names = names
    preds = []
    for img in imgs:
        opt.pred_img = img
        # make the prediction
        pred = detect(**vars(opt))
        preds.append(pred)
    # pdb.set_trace()
    return preds

if __name__ == "__main__":
    sample_img_path = "/home/nttung/person-in-context/BPA-Net/object_detector/sample.jpg"
    sample_img = cv2.imread(sample_img_path)
    imgs = [sample_img]
    example(imgs)
