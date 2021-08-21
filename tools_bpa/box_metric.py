import numpy as np
import copy
import cv2
import pdb
import math

def compute_IOU(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)

def get_center(rec):
    x_center = (rec[0] + rec[2])/2
    y_center = (rec[1] + rec[3])/2
    
    return (int(x_center), int(y_center))


def compute_intersect_over_smallest_area(rec1, rec2):
    # computing area of each rectangles
    
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / min(S_rec1, S_rec2)

def compute_area_ratio(box_numerator, box_denumerator):
    S_nume = (box_numerator[2] - box_numerator[0]) * (box_numerator[3] - box_numerator[1])
    S_denume = (box_denumerator[2] - box_denumerator[0]) * (box_denumerator[3] - box_denumerator[1])

    return S_nume/S_denume

def compute_euclid_distance(rec1, rec2):
    [y1_min, x1_min, y1_max, x1_max] = rec1
    [y2_min, x2_min, y2_max, x2_max] = rec2

    central_y1 = (y1_min + y1_max)/2
    central_x1 = (x1_min + x1_max)/2

    central_y2 = (y2_min + y2_max)/2
    central_x2 = (x2_min + x2_max)/2

    euclid_dis = math.sqrt((central_y1-central_y2)*(central_y1-central_y2)+(central_x1-central_x2)*(central_x1-central_x2))
    return euclid_dis

def vis_all_box_classes(img, list_boxes, list_classes, color):
    debug_img = copy.copy(img)

    for bbox, class_name in zip(list_boxes, list_classes):
        
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        debug_img = cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), color, thickness=2)
        debug_img = cv2.putText(debug_img, class_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)

    return debug_img

def join_box(rec1, rec2):
    [y1_min, x1_min, y1_max, x1_max] = rec1
    [y2_min, x2_min, y2_max, x2_max] = rec2

    y_min_big = min(y1_min, y2_min)
    x_min_big = min(x1_min, x2_min)
    y_max_big = max(y1_max, y2_max)
    x_max_big = max(x1_max, x2_max)

    big_rec = [y_min_big, x_min_big, y_max_big, x_max_big]
    return big_rec

def join_box_special(rec1, rec2):
    # rec1 human, rec2 phone
    # Special join only used for human and phone, drink box
    # with aim to join only the upper half of human 
    threshold_extend_ver = 20
    [y1_min, x1_min, y1_max, x1_max] = rec1
    [y2_min, x2_min, y2_max, x2_max] = rec2

    y_min_big = min(y1_min, y2_min)
    x_min_big = min(x1_min, x2_min)
    y_max_big = max(y1_max, y2_max)
    x_max_big =  min(x2_max + threshold_extend_ver, x1_max)# use extended vertical coordinate of phone box instead

    big_rec = [int(y_min_big), int(x_min_big), int(y_max_big), int(x_max_big)]
    return big_rec

def mask_phone_region(rec1, rec2, ori_img, file_name):
    # generate bbox specialized used for masking phone model
    ori_h, ori_w, _ = ori_img.shape
    threshold_extend_ver = 20
    threshold_extend = 20
    [y1_min, x1_min, y1_max, x1_max] = [int(rec1[0]), int(rec1[1]), int(rec1[2]), int(rec1[3])]
    [y2_min, x2_min, y2_max, x2_max] = [int(rec2[0]), int(rec2[1]), int(rec2[2]), int(rec2[3])]

    y_min_big = int(max(0, min(y1_min, y2_min) - threshold_extend))
    x_min_big = int(max(0, min(x1_min, x2_min) - threshold_extend))
    y_max_big = int(min(ori_w, max(y1_max, y2_max) + threshold_extend))
    x_max_big =  int(min(ori_h, x2_max + threshold_extend))# use extended vertical coordinate of phone box instead
    
    human_masked_phone_region = ori_img[x_min_big:x_max_big, y_min_big:y_max_big]
    phone_box = [y2_min - y_min_big, x2_min - x_min_big, 
                            y2_max - y_min_big, x2_max - x_min_big]
    mask_phone = np.zeros([phone_box[3] - phone_box[1], phone_box[2] - phone_box[0], 3], dtype=np.uint8)
    
    human_masked_phone_region[phone_box[1]:phone_box[3], phone_box[0]:phone_box[2]] = mask_phone
    return human_masked_phone_region