import os
import os.path as osp
import scipy.io as sio
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import time

def get_sorted_img_paths(seq_dir):
    img_path_list = []
    for file_name in os.listdir(seq_dir):
        if file_name.endswith('.jpg'):
            img_path_list.append(osp.join(seq_dir, file_name))
    # img_path_list.sort(key=lambda x : x.split('/')[-1])
    img_path_list.sort()
    
    return img_path_list
    
def get_mat_info(polygon_mat_path):
    polygon_mat = sio.loadmat(polygon_mat_path)
    # print(type(polygon_mat))
    # print(polygon_mat.keys())
    
    header = polygon_mat['__header__']
    version = polygon_mat['__version__']
    globals_ = polygon_mat['__globals__']
    polygons = polygon_mat['polygons']
    
    return header, version, globals_, polygons

def split_file_path(ori_path):
    seq_name, file_name = ori_path.split('/')[-2], ori_path.split('/')[-1] 
    return seq_name, file_name
    
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

def gen_yolo_format_data(sorted_img_paths, polygons, yolo_format_data_dir):
    # refresh the genearted data
    if os.path.exists(yolo_format_data_dir):
        shutil.rmtree(yolo_format_data_dir)
    os.mkdir(yolo_format_data_dir)
    yolo_img_dir = osp.join(yolo_format_data_dir, 'images')
    yolo_txt_dir = osp.join(yolo_format_data_dir, 'labels_hand_only')
    os.mkdir(yolo_img_dir)
    os.mkdir(yolo_txt_dir)
    
    polygons_100 = polygons[0]   # 100 polygons for all 100 frames in current folder
    pbar = tqdm(total=len(polygons_100))
    for img_path, polygon in zip(sorted_img_paths, polygons_100):
        time_s = time.time()
        img = cv2.imread(img_path)
        print('read img costs', time.time() - time_s)
        height, width, _ = img.shape
        own_left, own_right, other_left, other_right = polygon  # own_left hand, own_right hand, ...
        for hand in [own_left, own_right, other_left, other_right]:
            if hand.shape[1] == 0:
                continue
            
            MARGIN = 15    # enlarge the original bbox by MARGIN
            leftmost_x, rightmost_x = max(int(np.min(hand[:, 0]))- MARGIN, 0), min(int(np.max(hand[:, 0])) + MARGIN, width)
            top_most_y, bottom_most_y = max(int(np.min(hand[:, 1])) - MARGIN, 0), min(int(np.max(hand[:, 1])) + MARGIN, height)

            # generate labels as https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1 suggests
            # note that only hand labels are generated currently, and the labels of other objects will be generated by yolo and merged in.
            hand_cls_label = 0   # predefine the class of hand as 0
            x_center, y_center = (leftmost_x + rightmost_x) / 2, (top_most_y + bottom_most_y) / 2
            hand_width, hand_height = rightmost_x - leftmost_x, bottom_most_y - top_most_y
            normed_x_center, normed_y_center = float(x_center) / width, float(y_center) / height
            normed_hand_width, normed_hand_height = float(hand_width) / width, float(hand_height) / height
            
            seq_name, file_name = split_file_path(img_path)
            # yolo img 
            yolo_img_path = osp.join(yolo_img_dir, seq_name + '_' + file_name)
            shutil.copy(img_path, yolo_img_path)
            # yolo label txt
            yolo_txt_path = osp.join(yolo_txt_dir, seq_name + '_' + file_name.split('.jpg')[0] + '_hand_only.txt')
            with open(yolo_txt_path, 'w+') as res_txt:
                res_txt.write(str(hand_cls_label) + ' ' + str(normed_x_center) + ' ' + str(normed_y_center) + ' ' +
                            str(normed_hand_width) + ' ' + str(normed_hand_height) + '\n')
        pbar.update()
        
    pbar.close()
    
            
src_data_dir = '../_LABELLED_SAMPLES'
tmp_vis_sample_dir = '../exp_vis_samples'
yolo_format_data_dir = '../yolo_format_data'
for seq_name in os.listdir(src_data_dir):
    print(seq_name)
    seq_dir = osp.join(src_data_dir, seq_name)
    sorted_img_paths = get_sorted_img_paths(seq_dir)
    
    polygon_mat_path = osp.join(seq_dir, 'polygons.mat')
    header, version, globals_, polygons = get_mat_info(polygon_mat_path)
    # vis_polygon(sorted_img_paths, polygons, tmp_vis_sample_dir)
    # vis_bbox(sorted_img_paths, polygons, tmp_vis_sample_dir)
    gen_yolo_format_data(sorted_img_paths, polygons, yolo_format_data_dir)
    
    break