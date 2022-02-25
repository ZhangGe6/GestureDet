import os
import os.path as osp
import numpy as np
import json
import cv2
from tqdm import tqdm
# from .utils import get_bbox_from_joints

class EGOHandParser():
    def __init__(self, data_root):
        self.dataset_name = 'EGO'
        self.data_root = data_root
        self.raw_data_dir = osp.join(self.data_root, '_LABELLED_SAMPLES')
        self.hand_bbox_txt_dir = osp.join(self.data_root, 'hand_bboxes')
        self.objects_bbox_dir = osp.join(self.data_root, 'objects_bboxes')

        self.collect_samples()

    def collect_samples(self):
        self.samples = []
        for i, seq_name in enumerate(os.listdir(self.raw_data_dir)):
            print("collecting {}/{}: {} ...".format(i+1, len(os.listdir(self.raw_data_dir)), seq_name))
            for file_name in tqdm(os.listdir(os.path.join(self.raw_data_dir, seq_name))):
                if file_name.endswith('.jpg'):                
                    hand_bbox_txt_path = os.path.join(self.hand_bbox_txt_dir, seq_name, file_name.split('.jpg')[0] + '_hand_bboxes.txt')
                    # assert os.path.exists(hand_bbox_txt_path), hand_bbox_txt_path + " not exist!"

                    frame_path = os.path.join(self.raw_data_dir, seq_name, file_name)
                    height, width, _ = cv2.imread(frame_path).shape
                    
                    hand_bbox_txt_name = file_name.split('.jpg')[0] + '_hand_bboxes.txt'
                    hand_bbox_txt_path = osp.join(self.hand_bbox_txt_dir, seq_name, hand_bbox_txt_name)
                    # assert(os.path.exists(hand_bbox_txt_path))
                    if not os.path.exists(hand_bbox_txt_path):
                        hand_bbox = []
                    else:
                        hand_bbox = self.parse_bbox_txt(hand_bbox_txt_path)

                    object_bbox_txt_name = file_name.split('.jpg')[0] + '_objects_bboxes.txt'
                    object_bbox_txt_path = osp.join(self.objects_bbox_dir, seq_name, object_bbox_txt_name)
                    assert(os.path.exists(object_bbox_txt_path))
                    object_bbox = self.parse_bbox_txt(object_bbox_txt_path)

                    sample = dict()
                    sample['dataset_name'] = self.dataset_name
                    sample['file_name'] = file_name
                    sample['image_path'] = frame_path
                    sample['img_size'] = (height, width)
                    sample['hand_bbox'] = hand_bbox
                    sample['object_bbox'] = object_bbox
                    sample['joints'] = None
                    
                    self.samples.append(sample)
            # break
    
    def parse_bbox_txt(self, bbox_txt_path):
        bboxes = []
        with open(bbox_txt_path, 'r') as bbox_txt:
            for line in bbox_txt:
                bboxes.append(map(float, line.split())) # x, y, w, h, conf, class_
        
        return bboxes
        

if __name__ == '__main__':
    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/MHP_dataset'
    mhp_parser = MHPParser(data_root=data_root)
    print(len(mhp_parser.samples))

