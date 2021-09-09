import os
from typing_extensions import final 
import cv2
import pdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import json
import matplotlib.pyplot as plt
import wandb
import sys

sys.path.append("../")

# # object detector API
# sys.path.append("../object_detector")
from demo_script import detect_single_img

# pose extractor API 
sys.path.append("../external_lib")
from external_lib.demo_pose import *

# visualizer 
from tools_vis.vis_transparent import *


from torch.utils import data
from pathlib import Path
from tqdm import tqdm

from datasets.hoia_prediction import HOIA_infer as HOIA_infer_predict
from datasets.hoia_matching import HOIA_infer as HOIA_infer_matching 

from model.rel_prediction_network import SimpleNet
from model.bpa_network import PosePairAttenNet

from datasets.extract_image_features import get_vector

from tools_bpa.construct_co_matrix import co_mat_construction
from tools_bpa.utils import sigmoid_func

# ckpt
path_ckpt_predict = "../checkpoint/relation_prediction/simplenet_second_try_30_epoch.pth"
# path_ckpt_matching = "../checkpoint/pair_matching/pairnet_context_pose_attention_default_param_no_one_hot_59_epoch.pth"
path_ckpt_matching = "../checkpoint/pair_matching/BPA_17.pth"

# DETECTION RESULTS
path_json_detect = '/home/nttung/person-in-context/HOI-Det/HOI-A-new/test_2019.json'
# path_json_detect = '../detect_result/pred.json'
# path_json_detect = '../../hoia_test_2019_cascade.json'

# IMAGE FOLDER
path_image_folder = '/home/nttung/person-in-context/HOI-Det/HOI-A-new/test'

# PATH_JSON POSE
# path_json_pose = '/home/nttung/person-in-context/deep_experiment_v2/pose-data-2019/pose_results_test_yolo_2019.json'
path_json_pose = '/home/nttung/person-in-context/deep_experiment_v2/pose-data-2019/pose_results_test_gt_2019.json'
# path_json_pose = '/home/nttung/person-in-context/deep_experiment_v2/pose-data-2019/pose_results_test_cascade_2019.json'

# PATH JSON OUT

# path_json_out = './result/sim_net_pair_net_context_pose_attention_no_one_hot_59_epoch_prediction_test_2019_30_epoch.json'
# path_json_out = './result/gt_box_BPA_59_sim_30_test_2019.json'
path_json_out = '../result/gt_box_BPA_17_sim_30_test_2019.json'

def pair_matching_model(h_annots, o_annots, model_matching, device, img_path, kp_dict):
    '''
        o_annots: ["bbox", "category_id", "true_idx"]
    '''
    
    num_h = len(h_annots)
    num_o = len(o_annots)

    # empty matrix
    # sim_mat 
    sim_mat = np.zeros((num_h, num_o))
    # construct data for model pair matching
    # one instance: [hbox, obox, img_path]
    all_samples = []
    for idx_h, each_h in enumerate(h_annots):
        h_rec = each_h["bbox"]
        true_idx_h = each_h["true_idx"]

        pose_h = kp_dict[str(true_idx_h)]
        for idx_o, each_o in enumerate(o_annots):
            o_rec = each_o["bbox"]
            o_cat_id = each_o["category_id"]
            # add instance
            data_sample = {}
            data_sample["h_box"] = h_rec 
            data_sample["o_box"] = o_rec
            data_sample["object_category_id"] = int(o_cat_id)
            data_sample["kp_dict"] = pose_h

            data_sample["run_idx_h"] = idx_h
            data_sample["run_idx_o"] = idx_o
            all_samples.append(data_sample)

    # now create dataloader and start to infer
    test_data_matching = HOIA_infer_matching(all_samples, img_path)
    testloader_matching = torch.utils.data.DataLoader(test_data_matching, batch_size = 1, num_workers=2)
    
    atten_weights_dict = {}

    for i, data_test in enumerate(testloader_matching, 0):
            # construct input to model
            inputs, kv_dict, obj_query, run_idx_h, run_idx_o = data_test
            inputs = inputs.float().to(device)
            kv = kv_dict["keys_and_values"].float().to(device)
            obj_query = obj_query.float().to(device)

            output_score_match, atten_weights = model_matching.forward_return_attentional_map(inputs, obj_query, kv, kv)
            output_score_match = output_score_match.cpu().detach().numpy()
            run_idx_h = run_idx_h.cpu().detach().numpy()[0]
            run_idx_o = run_idx_o.cpu().detach().numpy()[0]

            sim_mat[run_idx_h][run_idx_o] = output_score_match
            atten_weights_dict[str(run_idx_h)+"_"+str(run_idx_o)] = atten_weights

    # now find relevant pairs
    pair_match = []
    for idx_h, each_h in enumerate(h_annots):
        proposed_o_idx_match = np.argmax(sim_mat[idx_h])

        # 1-1 mapping verify
        proposed_h_idx_match = np.argmax(sim_mat[:, proposed_o_idx_match])

        if idx_h == proposed_h_idx_match:

            # check if score smaller than threshold
            score_matching = sim_mat[proposed_h_idx_match][proposed_o_idx_match]
            
            chosen_o = o_annots[proposed_o_idx_match]
            true_idx_h = each_h["true_idx"]
            true_idx_o = chosen_o["true_idx"]
            pair_match.append((true_idx_h, true_idx_o, score_matching))

    return pair_match

def load_model_predict_matching():
    # set torch config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


    model_predict = SimpleNet(4116)
    model_predict.to(device)

    model_matching = PosePairAttenNet(6152, 2048 + 4, 17)
    model_matching.to(device)

    model_predict.load_state_dict(torch.load(path_ckpt_predict))
    model_matching.load_state_dict(torch.load(path_ckpt_matching))

    model_predict.eval()
    model_matching.eval()

    return model_predict, model_matching, device


def infer_single_video(path_frame_video):
    # load model predict
    print("============Loading BPA model==================")
    model_predict, model_matching, device = load_model_predict_matching()
    print("============Finish===============")
    
    # get width height of first image
    height, width, _ = cv2.imread(os.path.join(path_frame_video, os.listdir(path_frame_video)[0])).shape
    # prepare output video
    # get folder name of video 
    path_vid_out = path_frame_video.split("/")[-1]+ "_visualized"+".mp4"
    output_visualize = cv2.VideoWriter(path_vid_out, cv2.VideoWriter_fourcc(*'MP4V'), 5.0, (width, height))

    print("Running inference on video")
    for idx_run, path_image in enumerate(sorted(os.listdir(path_frame_video))):
        print("Num:", idx_run)
        # object detector results 
        path_image = os.path.join(path_frame_video, path_image)
        instance = detect_single_img(path_image)[0]
        if "annotations" not in instance:
            annot = instance["predictions"]
        else:
            annot = instance["annotations"]

        # filter human annotation
        human_boxes = []
        list_corres_idx_box = []
        
        for idx, each_annot in enumerate(annot):
            cat_id = each_annot["category_id"]
            format_box = each_annot["bbox"]

            # bottom left and top right
            format_box = [(format_box[0], format_box[3]), (format_box[2], format_box[1])]
            if int(cat_id) == 1: # human
                human_boxes.append(format_box)
                list_corres_idx_box.append(idx)


        # get pose information 
        kp_dict = run_single_pose(path_image, human_boxes, list_corres_idx_box, False)
        

        # filter human objects
        h_annot_list = []
        obj_group_category = {}
        for true_idx, each_annot in enumerate(annot):
            # group same objects first
            category_id = int(each_annot["category_id"])
            res = {}
            res["bbox"] = each_annot["bbox"]
            res["true_idx"] = true_idx
            if category_id == 1:
                # human 
                h_annot_list.append(res)

            else:
                # objects
                if category_id not in obj_group_category:
                    obj_group_category[category_id] = []

                res["category_id"] = category_id
                obj_group_category[category_id].append(res)


        # PAIRE MATCHING FLOW
        all_pair_res = []
        img_path = path_image
        for cat_id, obj_cat_list in obj_group_category.items():
            # find pair
            pair_res = pair_matching_model(h_annot_list, obj_cat_list, model_matching, device, img_path, kp_dict)
            
            # analyze_accept_pair_score.extend(list_accept_score)
            # analyze_reject_pair_score.extend(list_reject_score)
            all_pair_res.extend(pair_res)

        # PREDICTION FLOW
        triplet_result = []
        test_data = HOIA_infer_predict(all_pair_res, annot, img_path)
        testloader = torch.utils.data.DataLoader(test_data, batch_size = 1, num_workers=2)
        # for each pair, classify the correct relationship
        for i, data_test in enumerate(testloader, 0):
            # construct input to model
            input, obj_cat_id, s_id, o_id, score_matching = data_test
            input = input.float().to(device)

            output = model_predict(input)

            # get probability of the output with sigmoid function

            # get co-occurence relationship vector 
            co_vec = co_mat[obj_cat_id-1].reshape((1, co_mat.shape[1]))
            output = output.cpu().detach().numpy()
            score_matching = score_matching.cpu().detach().numpy()[0]
            # binarize the output
            filter_rel = co_vec*output
            final_rel_pos = np.where(filter_rel>0)

            for pos1, pos2 in zip(final_rel_pos[0], final_rel_pos[1]):
                score = filter_rel[pos1][pos2] 

                rel_cat_id = int(pos2+1)
                prob_rel = sigmoid_func(score)

                triplet = {}
                triplet["subject_id"] = s_id.item()
                triplet["object_id"] = o_id.item()

                if "score" in annot[s_id.item()]:
                    prob_subject = annot[s_id.item()]["score"]
                else:
                    prob_subject = 1

                if "score" in annot[s_id.item()]:
                    prob_object = annot[o_id.item()]["score"]
                else:
                    prob_object = 1

                triplet["category_id"] = rel_cat_id
                triplet["score"] = prob_subject * prob_object * prob_rel * score_matching
                triplet_result.append(triplet)
        
        img_read = cv2.imread(img_path)
        vis_img = draw_res_method(img_read, annot, triplet_result)
        output_visualize.write(vis_img)


    output_visualize.release()

        
    
    



if __name__ == '__main__':
    
    start_time = time.time()
    print("Constructing co-occurence matrix")
    path_json_construct_co_matrix = '/home/nttung/person-in-context/HOI-Det/HOI-A-new/train_2019.json'
    co_mat = co_mat_construction(path_json_construct_co_matrix)
    print("Finish constructing")
    
    path_frame_list = '/home/nttung/person-in-context/BPA-Net/video_demo/reading_book_harder_cut'

    infer_single_video(path_frame_list)

    print("Making inference in %s seconds" % (time.time() - start_time))
