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

path_json_gt = "/home/nttung/person-in-context/HOI-Det/HOI-A-new/test_2019.json"

# path_json_gt = "../result/yolo_box_BPA_17_sim_30_test_2019.json"
# path_json_gt = "../result/results_hoi_rule_base_with_yolo_boxes.json"

if os.path.isdir("./res_img_samples_annot_only") is False:
    os.mkdir("./res_img_samples_annot_only")



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
def pad_bottom(img, padding_size=30):
	h, w, c = img.shape
	canvas = np.zeros((h+padding_size, w, c))
	canvas[:h, :w] = img
	return canvas

def draw_res_method(img, annot, hoi_annot):
    # draw human-object bounding box for each hoi
    (h, w, _) = img.shape

    vis_img = copy.copy(img)
    # vis_img = pad_bottom(img, 300)

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
                score = ":" + str(each_hoi["score"])
        
        cat_name = rel_id2name[cat_id]

        h_box = annot[s_id]["bbox"]
        o_box = annot[o_id]["bbox"]

        h_center = get_center(h_box)
        o_center = get_center(o_box)

        ho_box = join_box(h_box, o_box)

        vis_img = cv2.rectangle(vis_img, (int(h_box[0]), int(h_box[1])), 
                                        (int(h_box[2]), int(h_box[3])),
                                        COLOR_LIST[0],
                                        3)
        
        vis_img = cv2.rectangle(vis_img, (int(o_box[0]), int(o_box[1])), 
                                        (int(o_box[2]), int(o_box[3])),
                                        COLOR_LIST[1],
                                        3)

    return vis_img
        

def visualize():

    # read sample image to be debugged 
    with open(path_json_sample_image, "r") as fp:
        json_img_name = json.load(fp)

    with open(path_json_gt, "r") as fp:
        json_gt = json.load(fp)

    json_img_name = json_img_name["3"]

    for instance in json_img_name:
        file_name = [*instance][0]
        print(file_name)

        json_data = list(filter(lambda k:k["file_name"] == file_name, json_gt))[0] 

        full_path_img = os.path.join(path_image_folder, file_name)  
        img_read = cv2.imread(full_path_img)



        if "annotations" in json_data:
            annot_1 = json_data["annotations"]
        else:
            annot_1 = json_data["predictions"]
        

        hoi_annot = json_data["hoi_annotation"]
        # draw result on each method
        vis_img = draw_res_method(img_read, annot_1, hoi_annot)

        # write results
        path_vis = os.path.join("./res_img_samples_annot_only", file_name)

        cv2.imwrite(path_vis, vis_img)




if __name__ == "__main__":
    visualize()
