import os
import time
import argparse
import copy
import cv2

import numpy as np
import torch.utils.data
import json
import pdb

from pathlib import Path
from tqdm import tqdm

import sys 
sys.path.append("../")
from datasets.extract_image_features import get_vector

object_id_list = ['background', 'person', 'cellphone', 'cigarette', 
                'drink', 'food', 'bicycle', 'motorcycle', 'horse', 'ball', 
                'computer', 'document']


rel_id_list = ['smoking', 'call', 'play(phone)', 'eat', 'drink',
                'ride', 'hold', 'kick', 'read', 'play(computer)']

def encode_onehot(cat_id):
    oh_encoding = np.zeros((1, len(object_id_list)))
    oh_encoding[:, cat_id] = 1
    return oh_encoding

def refine_box(h_box, o_box, img_path):
    h_box = list(map(int, h_box))
    o_box = list(map(int, o_box))

    img = cv2.imread(img_path)
    # localize human region
    # pdb.set_trace()
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
    # obj_zero_mask = np.zeros([norm_o_box[3] - norm_o_box[1], 
    #                                         norm_o_box[2] - norm_o_box[0], 3], dtype=np.uint8)
    # hom_region[norm_o_box[1]:norm_o_box[3], norm_o_box[0]:norm_o_box[2]] = obj_zero_mask

    # normalize bbox in range [0, 1]
    norm_h_box = [h_box[0]-ymin, h_box[1]-xmin, h_box[2]-ymin, h_box[3]-xmin]
    norm_o_box = [o_box[0]-ymin, o_box[1]-xmin, o_box[2]-ymin, o_box[3]-xmin]

    norm_out_h_box = [norm_h_box[0]/w_dim_region, norm_h_box[1]/h_dim_region, norm_h_box[2]/w_dim_region, norm_h_box[3]/h_dim_region]
    norm_out_o_box = [norm_o_box[0]/w_dim_region, norm_o_box[1]/h_dim_region, norm_o_box[2]/w_dim_region, norm_o_box[3]/h_dim_region]

    return norm_out_h_box, norm_out_o_box, h_region, o_region, hom_region

class HOIA(torch.utils.data.Dataset):
    def __init__(self, data_root, data_root_2, type):
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

        self.data_root_2 = data_root_2  # train test should be written carefully
        self.feature_cache = os.path.join(self.data_root_2, type, "feature_save")
        if os.path.isdir(self.feature_cache) is False:
            Path(self.feature_cache).mkdir(parents=True, exist_ok=True)

        with open(self.json_path, "r") as fp:
            data_json = json.load(fp)
        # create index of annotation
        self.data_list = []
        for i in range(len(data_json)):
            instance = data_json[i]

            f_name = instance["file_name"]
            annot = instance["annotations"]
            hoi_annot = instance["hoi_annotation"]

            for each_hoi in hoi_annot:
                info = {}
                s_id = each_hoi["subject_id"]
                o_id = each_hoi["object_id"]
                cat_rel_id = int(each_hoi["category_id"])

                # verify data
                if s_id >= len(annot) or o_id >= len(annot):
                    continue 

                s_box = list(map(int, annot[s_id]["bbox"]))
                o_box = list(map(int, annot[o_id]["bbox"]))
                o_category = int(annot[o_id]["category_id"])

                one_hot_category = encode_onehot(o_category)
                
                # create data to add to data list
                info["image_path"] = os.path.join(self.image_path, f_name)
                info["h_box"] = s_box
                info["o_box"] = o_box 
                info["feature_o"] = one_hot_category
                info["h_index"] = s_id
                info["o_index"] = o_id
                info["gt_label"] = cat_rel_id

                self.data_list.append(info)

        print("-------Finish initialization----------")
                

    def __getitem__(self, index):
        data_info = self.data_list[index]

        h_box = data_info["h_box"]
        o_box = data_info["o_box"]
        o_feat_one_hot = data_info["feature_o"]
        img_path = data_info["image_path"]
        cat_rel_id = data_info["gt_label"]

        norm_out_h_box, norm_out_o_box, h_region, o_region, hom_region = refine_box(h_box, o_box, img_path)

        # first check if feature of HUMAN and OBJECT, AND HUMAN OBJECT have been extracted
        path_feature = os.path.join(self.feature_cache, 'feature_{}.npy').format(index)
        exist = False
        if os.path.exists(path_feature):
            exist = True
        
        if exist:
            # automatically load feature from filen
            feature = np.load(path_feature)
            h_feat = feature[0].reshape((1, 2048))
            o_feat = feature[1].reshape((1, 2048))
            hom_feat = feature[1].reshape((1, 2048))

        else:
            # extract feature and save
            h_feat = get_vector(h_region)
            o_feat = get_vector(o_region)
            hom_feat = get_vector(hom_region)

            feat = np.concatenate((h_feat, o_feat, hom_feat))
            # save to npy array
            np.save(path_feature, feat)

        input = np.concatenate([norm_out_h_box, norm_out_o_box, list(h_feat[0]), list(o_feat[0]), list(hom_feat[0]), list(o_feat_one_hot[0])], 0)
        input = input.astype(np.float32)

        label = np.array(cat_rel_id-1) # decrease by one 
        
        return input, label


    def __len__(self):
        return len(self.data_list)


class HOIA_infer(torch.utils.data.Dataset):
    def __init__(self, pair_list, annot, img_path):
        self.img_path = img_path 
        self.data_infer = pair_list
        self.annot = annot

    def __getitem__(self, idx):
        data_pair = self.data_infer[idx]
        h_annot = self.annot[data_pair[0]]
        o_annot = self.annot[data_pair[1]]
        score_matching = data_pair[2]

        h_box = h_annot["bbox"]
        o_box = o_annot["bbox"]

        norm_h_box, norm_o_box, h_region, o_region, hom_region = refine_box(h_box, o_box, self.img_path)
        obj_cat_id = int(o_annot["category_id"])
        o_feat_one_hot = encode_onehot(obj_cat_id)


        h_feat = get_vector(h_region)
        o_feat = get_vector(o_region)
        hom_feat = get_vector(hom_region)

        input = np.concatenate([norm_h_box, norm_o_box, list(h_feat[0]), list(o_feat[0]), list(hom_feat[0]), list(o_feat_one_hot[0])])
        input = input.astype(np.float32)

        return input, obj_cat_id, data_pair[0], data_pair[1], score_matching

    def __len__(self):
        return len(self.data_infer)

def main():
    start_time = time.time()

    subset = ['train']
    data_root = '/home/nttung/person-in-context/HOI-Det/HOI-A-new'
    data_root_2 = '/home/nttung/person-in-context/deep_experiment_v2/auxilary-data'

    # Extract feature first
    for set_type in subset:
        data_set = HOIA(data_root, data_root_2, set_type)

        print('{} instances.'.format(len(data_set)))

        for index_data in tqdm(range(len(data_set))):
            input, label = data_set[index_data]
        print('Time elapsed: {:.2f}s'.format(time.time() - start_time))



if __name__ == '__main__':
    main()
