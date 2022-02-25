import os
import os.path as osp
import scipy.io as sio
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import time

def get_mat_info(polygon_mat_path):
    polygon_mat = sio.loadmat(polygon_mat_path)
    # print(type(polygon_mat))
    # print(polygon_mat.keys())
    
    header = polygon_mat['__header__']
    version = polygon_mat['__version__']
    globals_ = polygon_mat['__globals__']
    polygons = polygon_mat['polygons']
    
    return header, version, globals_, polygons

# def gen_hand_bboxes(sorted_img_paths, polygons, hand_bbox_dir):
def gen_hand_bboxes(seq_name, raw_data_dir, res_hand_bbox_dir):
    if osp.exists(osp.join(res_hand_bbox_dir, seq_name)):   # refresh
        shutil.rmtree(osp.join(res_hand_bbox_dir, seq_name))
    os.mkdir(osp.join(res_hand_bbox_dir, seq_name))

    polygon_mat_path = osp.join(raw_data_dir, seq_name, 'polygons.mat')
    polygons = get_mat_info(polygon_mat_path)[-1]
    polygons_100 = polygons[0]   # 100 polygons for all 100 frames in current folder

    pbar = tqdm(total=len(polygons_100))
    sorted_file_names = sorted([file for file in os.listdir(osp.join(raw_data_dir, seq_name)) if file.endswith('.jpg')])
    for file_name, polygon in zip(sorted_file_names, polygons_100):
        img_path = osp.join(raw_data_dir, seq_name, file_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        own_left, own_right, other_left, other_right = polygon  # own_left hand, own_right hand, ...
        for hand in [own_left, own_right, other_left, other_right]:
            if hand.shape[1] == 0:
                continue
            MARGIN = 15    # enlarge the original bbox by MARGIN
            x1, x2 = max(int(np.min(hand[:, 0]))- MARGIN, 0), min(int(np.max(hand[:, 0])) + MARGIN, width)
            y1, y2 = max(int(np.min(hand[:, 1])) - MARGIN, 0), min(int(np.max(hand[:, 1])) + MARGIN, height)

            # save as xywh format (COCO bbox format)
            res_hand_bbox_txt_path = osp.join(res_hand_bbox_dir, seq_name, file_name.split('.jpg')[0] + '_hand_bboxes.txt')
            with open(res_hand_bbox_txt_path, 'a+') as res_txt:
                res_txt.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

        pbar.update()
        
    pbar.close()
    
            
raw_data_dir = '../_LABELLED_SAMPLES'
res_hand_bbox_dir = '../hand_bboxes'
if not osp.exists(res_hand_bbox_dir):
    os.mkdir(res_hand_bbox_dir)

for seq_name in os.listdir(raw_data_dir):
    print(seq_name)
    gen_hand_bboxes(seq_name, raw_data_dir, res_hand_bbox_dir)
    
    # break