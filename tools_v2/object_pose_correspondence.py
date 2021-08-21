import os 
import numpy as np 
import json
import cv2
import math
import pdb

from tqdm import tqdm

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
    (187, 121, 133),
    (187, 121, 133),
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

object_id_list = ['background', 'person', 'cellphone', 'cigarette', 
                'drink', 'food', 'bicycle', 'motorcycle', 'horse', 'ball', 
                'computer', 'document']


rel_id_list = ['smoking', 'call', 'play(phone)', 'eat', 'drink',
                'ride', 'hold', 'kick', 'read', 'play(computer)']

num_kp = len(COCO_KEYPOINT_INDEXES.keys())
num_obj = len(object_id_list)
num_rel = len(rel_id_list)


def cal_spatial_correspondence_object_pose(pose_info, o_box_info):
    '''
        Return a vector of normalized score for correspondence with human pose
    '''
    # calculate distance from object center to kp
    [y1, x1] = [(o_box_info[0]+o_box_info[2])/2, (o_box_info[1]+o_box_info[3])/2]
    res_oj_kp_score = np.zeros((1, num_kp))

    list_kp_name = pose_info.keys()
    norm_val = 0.0

    for idx_kp, kp_name in enumerate(list_kp_name):
        [y2, x2] = pose_info[kp_name]
        euclid_dis = math.sqrt((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2))

        res_oj_kp_score[0][idx_kp] = euclid_dis

    pdb.set_trace()
    res = np.exp(-res_oj_kp_score)/np.sum(np.exp(-res_oj_kp_score))
    return res


def construct_obj_kp_correspondence(path_json_pose, path_json_data, path_img, path_json_out):
    # read json annot 
    with open(path_json_data, "r") as fp:
        data_json = json.load(fp)

    # read json pose 
    with open(path_json_pose, "r") as fp:
        data_json_pose = json.load(fp)


    corres_object_pose = np.zeros((num_obj, num_kp))

    for i in tqdm(range(len(data_json))):
        instance = data_json[i]
        file_name = instance["file_name"]
        annot = instance["annotations"]
        hoi_annot = instance["hoi_annotation"]

        # read info pose 
        pose_info = data_json_pose[file_name]

        for each_hoi in hoi_annot:
            s_id = each_hoi["subject_id"]
            o_id = each_hoi["object_id"]
            
            # construct correspondence for that human
            if s_id >= len(annot) or o_id >= len(annot) or str(s_id) not in pose_info:
                continue 

            pose_human_info = pose_info[str(s_id)]
            h_box_info = annot[s_id]["bbox"]
            o_box_info = annot[o_id]["bbox"]
            o_cat = annot[o_id]["category_id"]

            img_read = cv2.imread(os.path.join(path_img, file_name))
            cv2.imwrite("ts.png", img_read)
            
            res = cal_spatial_correspondence_object_pose(pose_human_info, o_box_info)
            pdb.set_trace()


if __name__ == "__main__":
    path_json_pose = '/home/nttung/person-in-context/deep_experiment_v2/pose-data-2019/pose_results_train_gt_2019.json'
    path_json_data = '/home/nttung/person-in-context/HOI-Det/HOI-A-new/train_2019.json'
    path_img = '/home/nttung/person-in-context/HOI-Det/HOI-A-new/trainval'
    path_json_out = './'
    construct_obj_kp_correspondence(path_json_pose, path_json_data, path_img, path_json_out)