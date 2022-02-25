import os
import os.path as osp
import scipy.io as sio
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import time

def vis_polygon(sorted_img_paths, polygons, tmp_save_dir):
    # print(type(polygons))   # <class 'numpy.ndarray'>
    # print(polygons.shape)   # (1, 100)
    # print(polygons[0].shape)    # (100,)
    # print(type(polygons[0][0]))  # <class 'numpy.void'>
    if not os.path.exists(tmp_save_dir):
        os.mkdir(tmp_save_dir)
    
    polygons_100 = polygons[0]   # 100 polygons for all 100 frames in current folder
    for i, (img_path, polygon) in enumerate(zip(sorted_img_paths, polygons_100)):
        img = cv2.imread(img_path)
        own_left, own_right, other_left, other_right = polygon  # own_left hand, own_right hand, ...
        for hand in [own_left, own_right, other_left, other_right]:
            if hand.shape[1] == 0:
                continue
            for point in hand:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=1)
                
        seq_name, file_name = split_file_path(img_path)
        save_path = osp.join(tmp_save_dir, 'vis_polygon_' + file_name)
        print("[INFO] saving to ", save_path)
        cv2.imwrite(save_path, img)

def vis_bbox(sorted_img_paths, polygons, tmp_save_dir):
    # print(type(polygons))   # <class 'numpy.ndarray'>
    # print(polygons.shape)   # (1, 100)
    # print(polygons[0].shape)    # (100,)
    # print(type(polygons[0][0]))  # <class 'numpy.void'>
    if not os.path.exists(tmp_save_dir):
        os.mkdir(tmp_save_dir)
    
    polygons_100 = polygons[0]   # 100 polygons for all 100 frames in current folder
    for i, (img_path, polygon) in enumerate(zip(sorted_img_paths, polygons_100)):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        own_left, own_right, other_left, other_right = polygon  # own_left hand, own_right hand, ...
        for hand in [own_left, own_right, other_left, other_right]:
            if hand.shape[1] == 0:
                continue
            
            MARGIN = 15    # enlarge the original bbox by MARGIN
            leftmost_x, rightmost_x = max(int(np.min(hand[:, 0]))- MARGIN, 0), min(int(np.max(hand[:, 0])) + MARGIN, width)
            top_most_y, bottom_most_y = max(int(np.min(hand[:, 1])) - MARGIN, 0), min(int(np.max(hand[:, 1])) + MARGIN, height)
            img = cv2.rectangle(img, (leftmost_x, top_most_y), (rightmost_x, bottom_most_y), color=(0, 0, 255), thickness=2)
            
        seq_name, file_name = split_file_path(img_path)
        save_path = osp.join(tmp_save_dir, 'vis_bbox_' + file_name)
        print("[INFO] saving to ", save_path)
        cv2.imwrite(save_path, img)

