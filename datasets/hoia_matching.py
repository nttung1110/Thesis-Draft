import os
import time
import argparse
import copy
import cv2

import numpy as np
import torch.utils.data
import json
import pdb

import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import sys 
sys.path.append("../")
from datasets.extract_image_features import get_vector

object_id_list = ['background', 'person', 'cellphone', 'cigarette', 
                'drink', 'food', 'bicycle', 'motorcycle', 'horse', 'ball', 
                'computer', 'document']


rel_id_list = ['smoking', 'call', 'play(phone)', 'eat', 'drink',
                'ride', 'hold', 'kick', 'read', 'play(computer)']

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

KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

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
offset_size = (16, 16) # size of 32


def encode_onehot(cat_id):
    oh_encoding = np.zeros((1, len(object_id_list)))
    oh_encoding[:, cat_id] = 1
    return oh_encoding

def gen_single_patch(coord_kp, img_w, img_h):
    [y_kp, x_kp] = coord_kp
    y_kp = max(0, y_kp)
    y_kp = min(img_w, y_kp)

    x_kp = max(0, x_kp)
    x_kp = min(img_h, x_kp)


    ymin = max(0, y_kp - offset_size[0])
    ymax = min(img_w, y_kp + offset_size[0])
    xmin = max(0, x_kp - offset_size[1])
    xmax = min(img_h, x_kp + offset_size[1])


    return [ymin, xmin, ymax, xmax]


def gen_patch_for_kp(img_w, img_h, kp_dict):
    patch_dict = {}
    for kp_name in kp_dict:
        coord_kp = kp_dict[kp_name]
        patch_dict[kp_name] = gen_single_patch(coord_kp, img_w, img_h)
        
    return patch_dict

def vis_patch_dict(patch_dict, img):
    vis_img = copy.copy(img)

    for kp_name in patch_dict:
        [ymin, xmin, ymax, xmax] = patch_dict[kp_name]["rectangle"]
        vis_img = cv2.rectangle(vis_img, (ymin, xmin), (ymax, xmax), (255, 0, 0), 3)
       
        patch_img = img[xmin:xmax, ymin:ymax]
        cv2.imwrite(kp_name+".jpg", patch_img)

    cv2.imwrite("vis.jpg", vis_img)

def refine_box_pose(h_box, o_box, kp_dict, img_path):
    h_box = list(map(int, h_box))
    o_box = list(map(int, o_box))

    img = cv2.imread(img_path)
    # localize human region
    clone_img = copy.copy(img)
    h_region = clone_img[h_box[1]:h_box[3], h_box[0]:h_box[2]]
    o_region = clone_img[o_box[1]:o_box[3], o_box[0]:o_box[2]]

    # extract human and object mask region

    ymin = min(h_box[0], o_box[0])
    ymax = max(h_box[2], o_box[2])
    xmin = min(h_box[1], o_box[1])
    xmax = max(h_box[3], o_box[3])

    # 2. localize region 
    clone_img = copy.copy(img)
    hom_region = clone_img[xmin:xmax, ymin:ymax]

    h_dim_region, w_dim_region, _ = hom_region.shape

    # 3. mask object
    norm_o_box = [o_box[0]-ymin, o_box[1]-xmin, o_box[2]-ymin, o_box[3]-xmin]

    # normalize bbox in range [0, 1]
    norm_h_box = [h_box[0]-ymin, h_box[1]-xmin, h_box[2]-ymin, h_box[3]-xmin]
    norm_out_h_box = [norm_h_box[0]/w_dim_region, norm_h_box[1]/h_dim_region, norm_h_box[2]/w_dim_region, norm_h_box[3]/h_dim_region]
    norm_out_o_box = [norm_o_box[0]/w_dim_region, norm_o_box[1]/h_dim_region, norm_o_box[2]/w_dim_region, norm_o_box[3]/h_dim_region]


    # get pose patch
    patch_dict = {}
    img_h, img_w, _ = img.shape
    for kp_name in kp_dict:
        coord_kp = kp_dict[kp_name]
        patch_dict[kp_name] = {} 
        patch_dict[kp_name]["rectangle"] = gen_single_patch(coord_kp, img_w, img_h)
        [ymin_pose, xmin_pose, ymax_pose, xmax_pose] = patch_dict[kp_name]["rectangle"]
        norm_kp = [ymin_pose-ymin, xmin_pose-xmin, ymax_pose-ymin, xmax_pose-xmin]
        norm_kp = [norm_kp[0]/w_dim_region, norm_kp[1]/h_dim_region, norm_kp[2]/w_dim_region, norm_kp[3]/h_dim_region]
        
        patch_dict[kp_name]["norm_rectangle"] = norm_kp

        patch_dict[kp_name]["region"] = img[xmin_pose:xmax_pose, ymin_pose:ymax_pose]

        # match with true idx of COCO idx kp
        patch_dict[kp_name]["idx_kp"] = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(kp_name)]
    
    return norm_out_h_box, norm_out_o_box, h_region, o_region, hom_region, patch_dict


class HOIA(torch.utils.data.Dataset):
    def __init__(self, data_root, data_root_2, data_root_pose, type, device):
        print("-------Initialize dataset----------")
        self.data_root = data_root
        if type == "train":
            json_name = "train_2019.json"
            img_fol_name = "trainval"
        elif type == "test":
            json_name = "test_2019.json"
            img_fol_name = "test"

        self.json_path = os.path.join(self.data_root, json_name)
        self.image_path = os.path.join(self.data_root, img_fol_name)
        self.device = device
        

        self.data_root_2 = data_root_2  # train test should be written carefully
        self.feature_cache = os.path.join(self.data_root_2, type, "feature_save")
        self.feature_pose_cache = os.path.join(self.data_root_2, type, "feature_pose_save")
        Path(self.feature_cache).mkdir(parents=True, exist_ok=True)
        Path(self.feature_pose_cache).mkdir(parents=True, exist_ok=True)

        # read json annot
        with open(self.json_path, "r") as fp:
            data_json = json.load(fp)
        
        # read pose annot
        with open(data_root_pose, "r") as fp:
            data_pose = json.load(fp)

        # create index of annotation
        self.data_list = []

        for i in range(len(data_json)):
            instance = data_json[i]

            f_name = instance["file_name"]
            annot = instance["annotations"]
            hoi_annot = instance["hoi_annotation"]
            all_obj_in_hoi = []
            human_object_list = {}
            positive_pair = []

            # get pose info 
            pose_info = data_pose[f_name]

            for each_hoi in hoi_annot:
                info = {}
                s_id = each_hoi["subject_id"]
                o_id = each_hoi["object_id"]
                cat_rel_id = int(each_hoi["category_id"])

                # verify data
                if s_id >= len(annot) or o_id >= len(annot) or str(s_id) not in pose_info:
                    continue    

                if s_id not in human_object_list:
                    human_object_list[s_id] = []

                human_object_list[s_id].append(o_id)
                all_obj_in_hoi.append(o_id)

                s_box = list(map(int, annot[s_id]["bbox"]))
                o_box = list(map(int, annot[o_id]["bbox"]))
                o_category = int(annot[o_id]["category_id"])

                one_hot_category = encode_onehot(o_category)
                
                # construct positive pair
                info["image_path"] = os.path.join(self.image_path, f_name)
                info["h_box"] = s_box
                info["o_box"] = o_box 
                info["feature_o"] = one_hot_category
                info["h_index"] = s_id
                info["o_index"] = o_id
                info["kp_dict"] = pose_info[str(s_id)]
                info["gt_label"] = 1

                positive_pair.append((s_id, o_id))
                self.data_list.append(info)

            # first filter human and object
            s_id_list = []
            o_id_list = []
            for idx_annot, each_annot in enumerate(annot):
                cat_id = int(each_annot["category_id"])
                if cat_id == 1:
                    # human
                    s_id_list.append(idx_annot)

                else:
                    # object
                    o_id_list.append(idx_annot)

            # finding negative pair
            for s_id in s_id_list:
                for o_id in o_id_list:
                    pair = (s_id, o_id)
                    if pair not in positive_pair:
                        info = {}

                        # construct negative pair
                        info["image_path"] = os.path.join(self.image_path, f_name)
                        info["h_box"] = annot[s_id]["bbox"]
                        info["o_box"] = annot[o_id]["bbox"]

                        cat_id = int(annot[o_id]["category_id"])
                        one_hot_category = encode_onehot(cat_id)
                        info["feature_o"] = one_hot_category
                        info["h_index"] = s_id
                        info["o_index"] = o_id
                        info["kp_dict"] = pose_info[str(s_id)]
                        info["gt_label"] = 0

                        self.data_list.append(info)
        print("-------Finish initialization----------")
                

    def __getitem__(self, index):
        data_info = self.data_list[index]

        h_box = data_info["h_box"]
        o_box = data_info["o_box"]
        o_feat_one_hot = data_info["feature_o"]
        img_path = data_info["image_path"]
        kp_dict = data_info["kp_dict"]
        is_match = data_info["gt_label"]

        norm_out_h_box, norm_out_o_box, h_region, o_region, hom_region, patch_dict = refine_box_pose(h_box, o_box, kp_dict, img_path)
        
        norm_out_h_box = torch.FloatTensor(norm_out_h_box).unsqueeze(0)
        norm_out_o_box = torch.FloatTensor(norm_out_o_box).unsqueeze(0)
        
        o_feat_one_hot = torch.FloatTensor(o_feat_one_hot)

        # first check if feature of HUMAN and HUMAN OBJECT MASK have been extracted
        path_feature = os.path.join(self.feature_cache, 'feature_matching_{}.pt').format(index)
        path_feature_pose = os.path.join(self.feature_pose_cache, 'feature_pose_matching_{}.pt').format(index)
        exist = False
        exist_pose = False
        if os.path.exists(path_feature):
            exist = True

        if os.path.exists(path_feature_pose):
            exist_pose = True
        
        if exist:
            # automatically load feature from file
            feature = torch.load(path_feature)
            h_feat = feature[0].reshape((1, 2048))
            o_feat = feature[1].reshape((1, 2048))
            hom_feat = feature[2].reshape((1, 2048))

        else:
            # extract feature and save
            h_feat = torch.from_numpy(get_vector(h_region))
            o_feat = torch.from_numpy(get_vector(o_region))
            hom_feat = torch.from_numpy(get_vector(hom_region))

            feat = torch.cat((h_feat, o_feat, hom_feat), dim=0)
            # save to npy array
            torch.save(feat, path_feature)

        if exist_pose:
            # automatically load feature pose from file
            feature_pose = torch.load(path_feature_pose)
            for kp_name in patch_dict:
                kp_info = patch_dict[kp_name]
                kp_true_idx = kp_info["idx_kp"]
                patch_dict[kp_name]["feature"] = feature_pose[kp_true_idx].reshape((1, 2048))

        else:
            # extract feature pose and save
            feat_pose = torch.empty((len(COCO_KEYPOINT_INDEXES), 2048))

            for kp_name in patch_dict:
                kp_info = patch_dict[kp_name]
                kp_true_idx = kp_info["idx_kp"]
                kp_region = kp_info["region"]

                patch_dict[kp_name]["feature"] = torch.from_numpy(get_vector(kp_region))
                feat_pose[kp_true_idx] = patch_dict[kp_name]["feature"]

                # save to npy array
            torch.save(feat_pose, path_feature_pose)

        fuse_info = torch.cat([norm_out_h_box, norm_out_o_box, h_feat, o_feat, hom_feat], dim=1)
        obj_query = torch.cat([norm_out_o_box, o_feat], dim=1)

        label = np.array(is_match) # decrease by one 

        # construct key and value list
        kv_dict = {}
        kv_dict["keys_and_values"] = torch.empty([len(COCO_KEYPOINT_INDEXES), 2048 + 4])

        for kp_name in patch_dict:
            kp_info = patch_dict[kp_name]
            norm_rec = torch.FloatTensor(kp_info["norm_rectangle"]).unsqueeze(0)
            res_kp_feat = kp_info["feature"]
            true_idx_kp = kp_info["idx_kp"]

            fuse_kp_info = torch.cat([norm_rec, res_kp_feat], dim=1)
            kv_dict["keys_and_values"][true_idx_kp] = fuse_kp_info


        return fuse_info, label, kv_dict, obj_query


    def __len__(self):
        return len(self.data_list)


class HOIA_infer(torch.utils.data.Dataset):
    def __init__(self, all_samples, img_path):
        self.img_path = img_path 
        self.all_samples = all_samples

    def __getitem__(self, idx):
        data_pair = self.all_samples[idx]

        h_box = data_pair["h_box"]
        o_box = data_pair["o_box"]
        kp_dict = data_pair["kp_dict"]
        
        run_idx_h = data_pair["run_idx_h"]
        run_idx_o = data_pair["run_idx_o"]

        norm_h_box, norm_o_box, h_region, o_region, hom_region, patch_dict = refine_box_pose(h_box, o_box, kp_dict, self.img_path)
        obj_cat_id = data_pair["object_category_id"]
        o_feat_one_hot = encode_onehot(obj_cat_id)

        norm_out_h_box = torch.FloatTensor(norm_h_box).unsqueeze(0)
        norm_out_o_box = torch.FloatTensor(norm_o_box).unsqueeze(0)
        
        o_feat_one_hot = torch.FloatTensor(o_feat_one_hot)


        h_feat = torch.from_numpy(get_vector(h_region))
        o_feat = torch.from_numpy(get_vector(o_region))
        hom_feat = torch.from_numpy(get_vector(hom_region))

        # input_model = input_model.astype(np.float32)
        feat_pose = torch.empty((len(COCO_KEYPOINT_INDEXES), 2048))

        for kp_name in patch_dict:
            kp_info = patch_dict[kp_name]
            kp_true_idx = kp_info["idx_kp"]
            kp_region = kp_info["region"]

            patch_dict[kp_name]["feature"] = torch.from_numpy(get_vector(kp_region))
            feat_pose[kp_true_idx] = patch_dict[kp_name]["feature"]


        fuse_info = torch.cat([norm_out_h_box, norm_out_o_box, h_feat, o_feat, hom_feat], dim=1)
        obj_query = torch.cat([norm_out_o_box, o_feat], dim=1)

        # construct key and value list
        kv_dict = {}
        kv_dict["keys_and_values"] = torch.empty([len(COCO_KEYPOINT_INDEXES), 2048 + 4])

        for kp_name in patch_dict:
            kp_info = patch_dict[kp_name]
            norm_rec = torch.FloatTensor(kp_info["norm_rectangle"]).unsqueeze(0)
            res_kp_feat = kp_info["feature"]
            true_idx_kp = kp_info["idx_kp"]

            fuse_kp_info = torch.cat([norm_rec, res_kp_feat], dim=1)
            kv_dict["keys_and_values"][true_idx_kp] = fuse_kp_info

        return fuse_info, kv_dict, obj_query, run_idx_h, run_idx_o

    def __len__(self):
        return len(self.all_samples)

def main():
    start_time = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    subset = ['train']
    data_root = '/home/nttung/person-in-context/HOI-Det/HOI-A-new'
    data_root_2 = '/home/nttung/person-in-context/deep_experiment_v2/test_matching_new'
    data_root_pose = '/home/nttung/person-in-context/deep_experiment_v2/pose-data-2019/pose_results_train_gt_2019.json'
    # Extract feature first
    for set_type in subset:
        data_set = HOIA(data_root, data_root_2, data_root_pose, set_type, device)

        print('{} instances.'.format(len(data_set)))

        for index_data in tqdm(range(len(data_set))):
            input, mask_pose, label = data_set[index_data]
        print('Time elapsed: {:.2f}s'.format(time.time() - start_time))



if __name__ == '__main__':
    main()
