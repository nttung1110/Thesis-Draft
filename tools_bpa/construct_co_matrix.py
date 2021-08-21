import numpy as np
import os
import cv2
import json
from torch.functional import norm
from tqdm import tqdm
import pdb

object_id_list = ['person', 'cellphone', 'cigarette', 
                'drink', 'food', 'bicycle', 'motorcycle', 'horse', 'ball', 
                'computer', 'document']


rel_id_list = ['smoking', 'call', 'play(phone)', 'eat', 'drink',
                'ride', 'hold', 'kick', 'read', 'play(computer)']

def co_mat_to_dict(co_mat):
    # easy for visualize
    vis_list = []
    for idx_obj in range(len(co_mat)):
        for idx_rel in range(len(co_mat[idx_obj])):
            name_rel = rel_id_list[idx_rel]
            name_obj = object_id_list[idx_obj]
            score = co_mat[idx_obj][idx_rel]
            vis_list.append((name_obj, name_rel, score))

    print(vis_list)

def binarize_co_mat(norm_co_mat):
    num_relation = 10
    num_object = 11
    threshold = 0.01
    # threshold = 0.1

    fil_pos = np.where(norm_co_mat>threshold)

    binarize_mat = np.zeros((num_object, num_relation))

    for idx_rel, idx_obj in zip(fil_pos[0], fil_pos[1]):
        binarize_mat[idx_rel][idx_obj] = 1

    return binarize_mat

def co_mat_construction(path_json_data):
    num_relation = 10
    num_object = 11


    # read data
    with open(path_json_data, "r") as fp:
        data = json.load(fp)

    # empty co matrix 
    co_mat = np.zeros((num_object, num_relation))

    for index in tqdm(range(len(data))):
        instance = data[index]
        annot = instance["annotations"]
        hoi_annot = instance["hoi_annotation"]
        
        for each_hoi in hoi_annot:
            o_id = int(each_hoi["object_id"])
            if o_id >= len(annot):
                continue

            category_obj = int(annot[o_id]["category_id"])
            category_hoi = int(each_hoi["category_id"])

            co_mat[category_obj-1][category_hoi-1] += 1 # be careful

    sum_column = np.sum(co_mat, axis=1)
    sum_column = sum_column.reshape((num_object, 1))
    norm_co_mat = co_mat/sum_column
    binarize_mat =  binarize_co_mat(norm_co_mat)
    # pdb.set_trace()
    return binarize_mat

if __name__ == "__main__":
    path_json = "/home/nttung/person-in-context/HOI-Det/HOI-A-new/train_2019.json"
    co_mat_construction(path_json)