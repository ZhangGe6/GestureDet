import numpy as np


def get_bbox_from_joints(joints, img_height, img_width, MARGIN=20):
    # 1. filter out some valid joints (like (0, 0) in cmu dataset) 
    # 2. enlarge the original bbox by MARGIN
    # joints_num = len(joints)
    # joints = [joint for joint in joints if 0 < joint[0] < img_width and 0 < joint[1] < img_height]
    # valid_joints_num = len(joints)
    # if not joints_num == valid_joints_num:
    #     # print("There are some joints out of the img. Skip this sample for more convincing training")
    #     return None
    joints = np.array(joints) 
    xmin, xmax = max(int(np.min(joints[:, 0]))- MARGIN, 0), min(int(np.max(joints[:, 0])) + MARGIN, img_width)
    ymin, ymax = max(int(np.min(joints[:, 1])) - MARGIN, 0), min(int(np.max(joints[:, 1])) + MARGIN, img_height)

    return [xmin, ymin, xmax, ymax]