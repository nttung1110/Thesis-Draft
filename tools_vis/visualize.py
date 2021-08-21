import os 
import json 
import pdb 
import sys 
import copy
import cv2
import numpy as np
from pathlib import Path


sys.path.append("../")

from tools_bpa.box_metric import join_box, get_center

path_image_folder = "/home/nttung/person-in-context/HOI-Det/HOI-A-new/test"
path_json_sample_image = "./image_samples/output.json"


if os.path.isdir("./res_img_samples") is False:
    os.mkdir("./res_img_samples")

path_image_res_method_1 = "./res_img_samples/method_1"
Path(path_image_res_method_1).mkdir(parents=True, exist_ok=True)

path_image_res_method_2 = "./res_img_samples/method_2"
Path(path_image_res_method_2).mkdir(parents=True, exist_ok=True)

path_image_res_both = "./res_img_samples/both_method"
Path(path_image_res_both).mkdir(parents=True, exist_ok=True)

path_json_method_1 = "../result/yolo_box_BPA_17_sim_30_test_2019.json"
path_json_method_2 = "../result/results_hoi_rule_base_with_yolo_boxes.json"


rel_id2name = ['None', 'smoking', 'call', 'play(phone)', 'eat', 'drink',
                'ride', 'hold', 'kick', 'read', 'play(computer)']

COLOR_LIST = [
    (255, 165, 0), #Orange
    (0, 255, 0), # GREEN
    (0, 0, 255), # RED
    (255, 255, 0), # YELLOW
    (255, 255, 0),
    (128, 128, 0),
    (0, 128, 128),
    (0, 128, 128),
    (0, 128, 128),
    (0, 128, 128),
    (0, 128, 128)
]
def pad_bottom(img, padding_size):
	h, w, c = img.shape
	canvas = np.ones((h+padding_size, w, c))*255
	canvas[:h, :w] = img
	return canvas

def draw_res_method(img, annot, hoi_annot):
    # draw human-object bounding box for each hoi
    (h, w, _) = img.shape

    vis_img = copy.copy(img)
    vis_img = pad_bottom(img, 600)

    color_assign = {}
    for (idx, each_hoi) in enumerate(hoi_annot):
        s_id = each_hoi["subject_id"]
        o_id = each_hoi["object_id"]
        cat_id = each_hoi["category_id"]
        
        key_for_color = str(s_id)+"_"+str(o_id)

        if str(s_id)+"_"+str(o_id) not in color_assign:
            color_assign[key_for_color] = COLOR_LIST[idx]

        chose_color = color_assign[key_for_color]

        score = ""
        if "score" in each_hoi:
            if str(each_hoi["score"]) != "0.99":
                score_round = round(each_hoi["score"], 4)
                score = ":" + str(score_round)
        
        cat_name = rel_id2name[cat_id]

        h_box = annot[s_id]["bbox"]
        o_box = annot[o_id]["bbox"]

        h_center = get_center(h_box)
        o_center = get_center(o_box)

        ho_box = join_box(h_box, o_box)

        # draw ho_box and matching line on image with color chosen
        vis_img = cv2.rectangle(vis_img, (int(ho_box[0]), int(ho_box[1])), 
                                        (int(ho_box[2]), int(ho_box[3])),
                                        chose_color,
                                        3)

        vis_img = cv2.line(vis_img, h_center, o_center, chose_color, 3)

        text_write = cat_name + score
        vis_img = cv2.putText(vis_img, text_write, (5, idx*55 + h+55),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                2, chose_color, 3, cv2.LINE_AA)
    return vis_img
        

def visualize():
    
    # read result method 1 
    with open(path_json_method_1, "r") as fp:
        json_1 = json.load(fp)

    # read result method 2
    with open(path_json_method_2, "r") as fp:
        json_2 = json.load(fp)

    # read sample image to be debugged 
    with open(path_json_sample_image, "r") as fp:
        json_img_name = json.load(fp)

    json_img_name = json_img_name["6"]

    for instance in json_img_name:
        file_name = [*instance][0]
        print(file_name)

        full_path_img = os.path.join(path_image_folder, file_name)  
        img_read = cv2.imread(full_path_img)

        method_1_data = list(filter(lambda k:k["file_name"] == file_name, json_1))[0] 
        method_2_data = list(filter(lambda k:k["file_name"] == file_name, json_2))[0]  


        if "annotations" in method_1_data:
            annot_1 = method_1_data["annotations"]
        else:
            annot_1 = method_1_data["predictions"]
        
        if "annotations" in method_2_data:
            annot_2 = method_2_data["annotations"]
        else:
            annot_2 = method_2_data["predictions"]

        hoi_annot_1 = method_1_data["hoi_annotation"]
        hoi_annot_2 = method_2_data["hoi_annotation"]
        # draw result on each method
        vis_img_1 = draw_res_method(img_read, annot_1, hoi_annot_1)
        vis_img_2 = draw_res_method(img_read, annot_2, hoi_annot_2)

        vis_img_both = cv2.hconcat([vis_img_1, vis_img_2])
        # write results
        path_vis_1 = os.path.join(path_image_res_method_1, file_name)
        path_vis_2 = os.path.join(path_image_res_method_2, file_name)
        path_vis_both = os.path.join(path_image_res_both, file_name)

        cv2.imwrite(path_vis_1, vis_img_1)
        cv2.imwrite(path_vis_2, vis_img_2)
        cv2.imwrite(path_vis_both, vis_img_both)




if __name__ == "__main__":
    visualize()
