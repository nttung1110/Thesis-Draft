import os 
import cv2 
import numpy as np 
import pdb
import json

import sys
sys.path.append("../")
from tools.box_metric import compute_IOU, compute_euclid_distance

def pair_matching(h_annots, o_annots, matching_method):
    '''
        o_annots: ["bbox", "category_id", "true_idx"]
    '''
    

    num_h = len(h_annots)
    num_o = len(o_annots)

    # empty matrix
    # sim_mat 
    sim_mat = np.zeros((num_h, num_o))

    for idx_h, each_h in enumerate(h_annots):
        h_rec = each_h["bbox"]
        for idx_o, each_o in enumerate(o_annots):
            o_rec = each_o["bbox"]
            if matching_method == "iou_intersection":
                sim_score = compute_IOU(h_rec, o_rec)
            elif matching_method == "euclidean":
                sim_score = compute_euclid_distance(h_rec, o_rec)
            sim_mat[idx_h][idx_o] = sim_score

    # finding match pairs for each human
    pair_match = []
    if matching_method == "iou_intersection":
        for idx_h, each_h in enumerate(h_annots):
            proposed_o_idx_match = np.argmax(sim_mat[idx_h])
            if sim_mat[idx_h][proposed_o_idx_match] == 0:
                continue

            # 1-1 mapping verify
            proposed_h_idx_match = np.argmax(sim_mat[:, proposed_o_idx_match])

            if idx_h == proposed_h_idx_match:
                chosen_o = o_annots[proposed_o_idx_match]
                true_idx_h = each_h["true_idx"]
                true_idx_o = chosen_o["true_idx"]
                pair_match.append((true_idx_h, true_idx_o))
    

    elif matching_method == "euclidean":
        for idx_h, each_h in enumerate(h_annots):
            proposed_o_idx_match = np.argmin(sim_mat[idx_h])

            # 1-1 mapping verify
            proposed_h_idx_match = np.argmin(sim_mat[:, proposed_o_idx_match])
        
            if idx_h == proposed_h_idx_match:
                chosen_o = o_annots[proposed_o_idx_match]
                true_idx_h = each_h["true_idx"]
                true_idx_o = chosen_o["true_idx"]
                pair_match.append((true_idx_h, true_idx_o))
    

    return pair_match

# def test():

def pair_matching_debug(h_annots, o_annots, matching_method):
    '''
        o_annots: ["bbox", "category_id", "true_idx"]
    '''
    

    num_h = len(h_annots)
    num_o = len(o_annots)

    # empty matrix
    # sim_mat 
    sim_mat = np.zeros((num_h, num_o))

    for idx_h, each_h in enumerate(h_annots):
        h_rec = each_h["bbox"]
        for idx_o, each_o in enumerate(o_annots):
            o_rec = each_o["bbox"]
            if matching_method == "iou_intersection":
                sim_score = compute_IOU(h_rec, o_rec)
            elif matching_method == "euclidean":
                sim_score = compute_euclid_distance(h_rec, o_rec)
            sim_mat[idx_h][idx_o] = sim_score

    # finding match pairs for each human
    pair_match = []
    if matching_method == "iou_intersection":
        for idx_h, each_h in enumerate(h_annots):
            proposed_o_idx_match = np.argmax(sim_mat[idx_h])
            if sim_mat[idx_h][proposed_o_idx_match] == 0:
                continue

            # 1-1 mapping verify
            proposed_h_idx_match = np.argmax(sim_mat[:, proposed_o_idx_match])

            if idx_h == proposed_h_idx_match:
                chosen_o = o_annots[proposed_o_idx_match]
                true_idx_h = each_h["true_idx"]
                true_idx_o = chosen_o["true_idx"]
                pair_match.append((true_idx_h, true_idx_o))
    

    elif matching_method == "euclidean":
        for idx_h, each_h in enumerate(h_annots):
            proposed_o_idx_match = np.argmin(sim_mat[idx_h])

            # 1-1 mapping verify
            proposed_h_idx_match = np.argmin(sim_mat[:, proposed_o_idx_match])
        
            if idx_h == proposed_h_idx_match:
                chosen_o = o_annots[proposed_o_idx_match]
                true_idx_h = each_h["true_idx"]
                true_idx_o = chosen_o["true_idx"]
                pair_match.append((true_idx_h, true_idx_o))
    
    pdb.set_trace()

    return pair_match

def test():
    path_json = "../result/sim_net_prediction_test_2019_30_epoch_euclid_matching.json"

    with open(path_json, "r") as fp:
        data_json = json.load(fp)

    for instance in data_json :
        file_name = instance["file_name"]
        if file_name != "test_006578.jpg":
            continue

        all_annot = instance["annotations"]

        h_annot = []
        o_annot = []
        for true_idx, each_annot in enumerate(all_annot):
            box = each_annot["bbox"]
            cat_id = int(each_annot["category_id"])
            
            tmp_dict = {}
            tmp_dict["true_idx"] = true_idx 
            tmp_dict["bbox"] = box 
            if cat_id == 1:
                h_annot.append(tmp_dict)
            
            else:
                o_annot.append(tmp_dict)

        pair_matching_debug(h_annot, o_annot, "euclidean")

        

if __name__ == '__main__':
    test()