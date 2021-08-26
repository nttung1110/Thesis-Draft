import json 
import cv2
import os
import pdb

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

def vis_image_pose(image_name):
    path_img_in = "../../../../../HOI-Det/HOI-A-new/Train_2021"
    path_pose_json = "./pose_results_train_detect_gt_2021.json"
    # read json pose
    with open(path_pose_json, "r") as fp :
        data = json.load(fp)
    img_read = cv2.imread(os.path.join(path_img_in, image_name))
    pose_info = data[image_name]

    for idx_human, pose_human  in pose_info.items():
        # Draw each point on image
        for idx_kp, kp_name in enumerate(pose_human):
            coord = pose_human[kp_name]
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(img_read, (x_coord, y_coord), 4, COLOR_PANEL[idx_kp], 2)

    path_out_vis = "./vis_debug"
    if os.path.isdir(path_out_vis) is False:
        os.mkdir(path_out_vis)

    img_out = os.path.join(path_out_vis, image_name)
    cv2.imwrite(img_out, img_read)

if __name__ == "__main__":
    pdb.set_trace()