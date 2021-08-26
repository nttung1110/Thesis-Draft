from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import json
import numpy as np
import copy
import pdb 

import sys
sys.path.append("/home/nttung/person-in-context/BPA-Net/external_lib/sota_pose_estimation/deep-high-resolution-net.pytorch/lib")
sys.path.append("/home/nttung/person-in-context/BPA-Net/external_lib/sota_pose_estimation/deep-high-resolution-net.pytorch/")
import time

# import pdb 
# pdb.set_trace()
# import _init_paths
import models

pdb.set_trace()
from config import cfg
from config import update_config
from core.inference import get_final_preds
from lib.utils.transforms import get_affine_transform


CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COLOR_PANEL = [
    (255, 0, 0), #BLUE
    (0, 255, 0), # GREEN
    (0, 255, 0), 
    (0, 0, 255), # RED
    (0, 0, 255),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (128, 128, 0),
    (128, 128, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (0, 128, 128),
    (0, 128, 128)
]

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def get_single_pose_api(img, human_bbox, pose_transform, pose_model, cfg):
    # transformation
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    format_box = [(human_bbox[0], human_bbox[3]), (human_bbox[2], human_bbox[1])]
    center, scale = box_to_center_scale(format_box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    centers = [center]
    scales = [scale]
    
    pose_preds = get_pose_estimation_prediction(pose_model, img, centers, scales, transform=pose_transform)
    if len(pose_preds) == 0:
        return None
    else:
        return pose_preds[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    # parser.add_argument('--cfg', type=str, required=True)
    # parser.add_argument('--input_folder', type=str, default='../../../../HOI-Det/HOI-A-new/trainval')
    # parser.add_argument('--input_json', type=str, default='../../../../HOI-Det/HOI-A-new/train_2019.json')
    
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def run_json_pose(path_img_in, path_json_detect, path_json_out_pose, IS_VIS):
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    

    # hard code args
    args = parse_args()
    args.cfg = "config_infer_pose.yaml"
    # model checkpoint has been encoded inside yaml
    # args = parse_args()
    update_config(cfg, args)

    pdb.set_trace()

    # get pose model
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()
    num = 0 
    
    # read json data 
    with open(path_json_detect, "r") as fp:
        data = json.load(fp)
    
    out_json = {}

    for instance in data:
        print("Num:", num)

        img_name = instance["file_name"]

        if "annotations" not in instance:
            annot = instance["predictions"]
        else:
            annot = instance["annotations"]

        out_json[img_name] = {}

        img_read_name = os.path.join(path_img_in, img_name)
        img_read = cv2.imread(img_read_name)
        _, img_w, img_h = img_read.shape
        # get human boxes  

        human_boxes = []
        list_corres_idx_box = []
        
        for idx, each_annot in enumerate(annot):
            cat_id = each_annot["category_id"]
            format_box = each_annot["bbox"]

            # bottom left and top right
            format_box = [(format_box[0], format_box[3]), (format_box[2], format_box[1])]
            if cat_id == "1" or cat_id == 1: # human
                human_boxes.append(format_box)
                list_corres_idx_box.append(idx)

        centers = []
        scales = []
        for box in human_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        list_pose_preds = []
        for center, scale in zip(centers, scales):
            input_centers = [center]
            input_scales = [scale]
            pose_preds = get_pose_estimation_prediction(pose_model, img_read, input_centers, input_scales, transform=pose_transform)
        
            if len(pose_preds) == 0:
                list_pose_preds.append([])

            else:
                list_pose_preds.append(pose_preds[0])

        for idx_h_box, pose_preds in zip(list_corres_idx_box, list_pose_preds):
            
            if len(pose_preds) == 0:
                # pdb.set_trace()
                continue

            coords_info = {}
            for idx_kp, coord in enumerate(pose_preds):
                x_coord, y_coord = int(coord[0]), int(coord[1])
                name_kp = COCO_KEYPOINT_INDEXES[idx_kp]
                coords_info[name_kp] = [x_coord, y_coord]

            out_json[img_name][idx_h_box] = coords_info
        
        if IS_VIS:
            path_vis = "./vis_pose"
            image_debug = copy.copy(img_read)
            for coords in pose_preds:
                # Draw each point on image
                for idx_kp, coord in enumerate(coords):
                    x_coord, y_coord = int(coord[0]), int(coord[1])
                    cv2.circle(image_debug, (x_coord, y_coord), 4, COLOR_PANEL[idx_kp], 2)

                # write vis ouput
                if os.path.isdir(path_vis) is False:
                    os.mkdir(path_vis)
                
                # pdb.set_trace()
                vis_path = os.path.join(path_vis, img_name)
                cv2.imwrite(vis_path, image_debug)
        
        
        num += 1
        # if num == 1000:
        #     break

    with open(path_json_out_pose, "w") as fp:
        json.dump(out_json, fp, indent=4)

def run_single_pose(path_img_in, path_json_detect, path_json_out_pose, IS_VIS):
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    

    # hard code args
    args = parse_args()
    args.cfg = "config_infer_pose.yaml"
    # model checkpoint has been encoded inside yaml
    # args = parse_args()
    update_config(cfg, args)

    pdb.set_trace()

    # get pose model
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()
    num = 0 
    
    # read json data 
    with open(path_json_detect, "r") as fp:
        data = json.load(fp)
    
    out_json = {}

    for instance in data:
        print("Num:", num)

        img_name = instance["file_name"]

        if "annotations" not in instance:
            annot = instance["predictions"]
        else:
            annot = instance["annotations"]

        out_json[img_name] = {}

        img_read_name = os.path.join(path_img_in, img_name)
        img_read = cv2.imread(img_read_name)
        _, img_w, img_h = img_read.shape
        # get human boxes  

        human_boxes = []
        list_corres_idx_box = []
        
        for idx, each_annot in enumerate(annot):
            cat_id = each_annot["category_id"]
            format_box = each_annot["bbox"]

            # bottom left and top right
            format_box = [(format_box[0], format_box[3]), (format_box[2], format_box[1])]
            if cat_id == "1" or cat_id == 1: # human
                human_boxes.append(format_box)
                list_corres_idx_box.append(idx)

        centers = []
        scales = []
        for box in human_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        list_pose_preds = []
        for center, scale in zip(centers, scales):
            input_centers = [center]
            input_scales = [scale]
            pose_preds = get_pose_estimation_prediction(pose_model, img_read, input_centers, input_scales, transform=pose_transform)
        
            if len(pose_preds) == 0:
                list_pose_preds.append([])

            else:
                list_pose_preds.append(pose_preds[0])

        for idx_h_box, pose_preds in zip(list_corres_idx_box, list_pose_preds):
            
            if len(pose_preds) == 0:
                # pdb.set_trace()
                continue

            coords_info = {}
            for idx_kp, coord in enumerate(pose_preds):
                x_coord, y_coord = int(coord[0]), int(coord[1])
                name_kp = COCO_KEYPOINT_INDEXES[idx_kp]
                coords_info[name_kp] = [x_coord, y_coord]

            out_json[img_name][idx_h_box] = coords_info
        
        if IS_VIS:
            path_vis = "./vis_pose"
            image_debug = copy.copy(img_read)
            for coords in pose_preds:
                # Draw each point on image
                for idx_kp, coord in enumerate(coords):
                    x_coord, y_coord = int(coord[0]), int(coord[1])
                    cv2.circle(image_debug, (x_coord, y_coord), 4, COLOR_PANEL[idx_kp], 2)

                # write vis ouput
                if os.path.isdir(path_vis) is False:
                    os.mkdir(path_vis)
                
                # pdb.set_trace()
                vis_path = os.path.join(path_vis, img_name)
                cv2.imwrite(vis_path, image_debug)
        
        
        num += 1
        # if num == 1000:
        #     break

    with open(path_json_out_pose, "w") as fp:
        json.dump(out_json, fp, indent=4)

def vis_image_pose(image_name):
    path_img_in = "../../../../../HOI-Det/HOI-A-new/trainval"
    path_pose_json = "./pose_results_train_update.json"
    # read json pose
    with open(path_pose_json, "r") as fp :
        data = json.load(fp)

    img_read = cv2.imread(os.path.join(path_img_in, image_name))
    pose_info = data[image_name]

    for idx_pose, pose_coords in pose_info.values():
        for pose_name, coords in pose_coords.values():
            # Draw each point on image
            for idx_kp, coord in enumerate(coords):
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(img_read, (x_coord, y_coord), 4, COLOR_PANEL[idx_kp], 2)

    path_out_vis = "./vis_debug"
    if os.path.isdir(path_out_vis) is False:
        os.mkdir(path_out_vis)

    img_out = os.path.join(path_out_vis, image_name)
    cv2.imwrite(img_out, img_read)


if __name__ == '__main__':
    # path_img_in = "../../../../../HOI-Det/HOI-A-new/test"
    # # path_json_detect = "../../../detect_all/detect_results_train_update.json"
    # path_json_detect = "../../../../../HOI-Det/HOI-A-new/test_2019.json"
    # path_json_out_pose = "./pose_results_test_update_detect_gt.json"

    # path_img_in_train = "../../../../../HOI-Det/HOI-A-new/Train_2021"
    # path_json_detect_train = "../../../detect_all/detect_50k_epoch_results_train_2021.json"
    # # path_json_detect_train = "../../../../../HOI-Det/HOI-A-new/train_2021.json"
    # path_json_out_pose_train = "./pose_results_train_detect_50k_epoch_2021.json"

    # path_img_in_test = "../../../../../HOI-Det/HOI-A-new/Test_2021"
    # path_json_detect_test = "../../../detect_all/detect_50k_epoch_results_test_2021.json"
    # path_json_out_pose_test = "./pose_results_test_detect_50k_epoch_2021.json"

    # path_img_in_train = "../../../../../HOI-Det/HOI-A-new/Train_2021"
    # path_json_detect_train = "../../../detect_all/detect_hybrid_results_train_2021.json"
    # # path_json_detect_train = "../../../../../HOI-Det/HOI-A-new/train_2021.json"
    # path_json_out_pose_train = "./pose_results_train_detect_hybrid_2021.json"

    # path_img_in_test = "../../../../../HOI-Det/HOI-A-new/Test_2021"
    # # path_json_detect_train = "../../../detect_all/detect_results_train_update.json"
    # path_json_detect_test = "../../../detect_all/detect_hybrid_results_test_2021.json"
    # path_json_out_pose_test = "./pose_results_test_detect_hybrid_2021.json"
    
    # path_img_in_train = "../../../../../HOI-Det/HOI-A-new/Train_2021"
    # path_json_detect_train = "../../../detect_all/detect_70k_epoch_results_train_2021.json"
    # # path_json_detect_train = "../../../../../HOI-Det/HOI-A-new/train_2021.json"
    # path_json_out_pose_train = "./pose_results_train_detect_70k_epoch_2021.json"

    # path_img_in_test = "../../../../../HOI-Det/HOI-A-new/Test_2021"
    # path_json_detect_test = "../../../detect_all/detect_70k_epoch_results_test_2021.json"
    # path_json_out_pose_test = "./pose_results_test_detect_70k_epoch_2021.json"

    # path_img_in_train = "../../../../../HOI-Det/HOI-A-new/trainval"
    # path_json_detect_train = "../../../detect_all/detect_results_train_2019_4_6.json"
    # # path_json_detect_train = "../../../../../HOI-Det/HOI-A-new/train_2021.json"
    # path_json_out_pose_train = "./pose_results_trainval_detect_50k_epoch_2019.json"

    # path_img_in_train = "../../../../../HOI-Det/HOI-A-new/trainval"
    # path_json_detect_train = "../../../../../HOI-Det/HOI-A-new/train_2019.json"
    # path_json_out_pose_train = "./pose_results_train_gt_2019.json"
    
    # path_img_in_test = "../../../../../HOI-Det/HOI-A-new/test"
    # path_json_detect_test = "../../../../../HOI-Det/HOI-A-new/test_2019.json"
    # path_json_out_pose_test = "./pose_results_test_gt_2019.json"

    path_img_in_test = "/home/nttung/person-in-context/HOI-Det/HOI-A-new/test"
    # path_json_detect_test = "../../asnet-test2019-pred.json"
    # path_json_detect_test = '/home/nttung/person-in-context/deep_experiment_v2/detect_result/pred.json'

    # path_json_detect_test = '/home/nttung/person-in-context/hoia_test_2019_cascade.json'
    path_json_detect_test = '/home/nttung/person-in-context/cascade_r152_tta_preds.json'
    path_json_out_pose_test = "/home/nttung/person-in-context/pose-data-2019/pose_results_test_cascade_tta_2019.json"

    IS_VIS = False
    # train data 
    # run_json_pose(path_img_in_train, path_json_detect_train, path_json_out_pose_train, IS_VIS)

    # # test_data
    run_json_pose(path_img_in_test, path_json_detect_test, path_json_out_pose_test, IS_VIS)
