import os
import os.path as osp
import numpy as np
import json
import cv2
from tqdm import tqdm
# from .utils import get_bbox_from_joints

class SelfCollectedParser():
    def __init__(self, data_root, skip_no_hand_sample=False):
        self.dataset_name = 'self_collected'
        self.anno_path = data_root
        self.skip_no_hand_sample = skip_no_hand_sample

        self.collect_samples()

    def collect_samples(self):
        with open(self.anno_path, 'r') as f:
            anno = json.load(f)
        json_samples = anno['samples']
    
        self.samples = []
        for i, json_sample in enumerate(json_samples):
            print("collecting {}/{} ...".format(i+1, len(json_samples)))
            frame_path = json_sample['frame_path']
            hands = json_sample['hands']
            object_bboxes = json_sample['object_bboxes']
            height, width, _ = cv2.imread(frame_path).shape

            sample = dict()
            sample['dataset_name'] = self.dataset_name
            sample['image_path'] = frame_path
            sample['img_size'] = (height, width)
            sample['object_bbox'] = object_bboxes
            if len(hands) > 0:
                sample['hand_bbox'] = self.hands_joint2bboxes(hands[0], width, height)
                sample['joints'] = hands[0]
            else:
                sample['hand_bbox'] = []
                sample['joints'] = []
            
            self.samples.append(sample)
        # break
    
    def hands_joint2bboxes(self, hand, width, height, MARGIN=80):
        # 1. suppose 1 hand per image
        # 2. output xyxy format
        hand = np.array(hand)
        x1, x2 = max(int(np.min(hand[:, 0]))- MARGIN, 0), min(int(np.max(hand[:, 0])) + MARGIN, width)
        y1, y2 = max(int(np.min(hand[:, 1])) - MARGIN, 0), min(int(np.max(hand[:, 1])) + MARGIN, height)
        return [x1, y1, x2, y2]




    