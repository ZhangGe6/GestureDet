import os
import os.path as osp
import numpy as np
import json
import cv2
from tqdm import tqdm
# from .utils import get_bbox_from_joints

class MHPParser():
    def __init__(self, data_root):
        self.dataset_name = 'MHP'
        self.data_root = data_root
        self.frames_dir = osp.join(self.data_root, 'annotated_frames')
        self.hand_bbox_txt_dir = osp.join(self.data_root, 'bounding_boxes')
        self.objects_bbox_dir = osp.join(self.data_root, 'objects_bounding_boxes')
        self.joint_txt_dir = osp.join(self.data_root, 'projections_2d')

        self.collect_samples()

    def collect_samples(self):
        self.samples = []
        for i, data_id in enumerate(os.listdir(self.frames_dir)):
            print("collecting {}/{}: {} ...".format(i+1, len(os.listdir(self.frames_dir)), data_id))
            for file_name in tqdm(os.listdir(osp.join(self.frames_dir, data_id))):
                if file_name.endswith('.jpg'):
                    frame_id, camera_id = file_name.split('.jpg')[0].split('_webcam_')
                    if camera_id == '4': continue  # bad label, skip

                    sample = dict()
                    frame_path = osp.join(self.frames_dir, data_id, file_name)
                    height, width, _ = cv2.imread(frame_path).shape
                    
                    hand_bbox_txt_name = frame_id + '_bbox_' + camera_id + '.txt'
                    hand_bbox_txt_path = osp.join(self.hand_bbox_txt_dir, data_id, hand_bbox_txt_name)
                    assert(os.path.exists(hand_bbox_txt_path))
                    hand_bbox = self.parse_hand_bbox_txt(hand_bbox_txt_path)

                    object_bbox_txt_name = frame_id + '_objects_bbox_' + camera_id + '.txt'
                    object_bbox_txt_path = osp.join(self.objects_bbox_dir, data_id, object_bbox_txt_name)
                    assert(os.path.exists(object_bbox_txt_path))
                    object_bbox = self.parse_object_bbox_txt(object_bbox_txt_path)

                    joint_txt_name = frame_id + '_jointsCam_' + camera_id + '.txt'
                    joint_txt_path = osp.join(self.joint_txt_dir, data_id, joint_txt_name)
                    assert(os.path.exists(joint_txt_path))
                    joints = self.parse_joints_txt(joint_txt_path)

                    sample['dataset_name'] = self.dataset_name
                    sample['file_name'] = file_name
                    sample['image_path'] = frame_path
                    sample['img_size'] = (height, width)
                    sample['hand_bbox'] = hand_bbox
                    sample['object_bbox'] = object_bbox
                    sample['joints'] = joints
                    
                    self.samples.append(sample)
            # break

    
    def parse_hand_bbox_txt(self, bbox_txt_path):
        # parse to xyxy
        def tlbr2xyxy(tlbr):
            # tlbr: ymin, xmin, ymax, xmax
            # xyxy: xmin, ymin, xmax, ymax
            xyxy = []
            xyxy.append(tlbr[1])
            xyxy.append(tlbr[0])
            xyxy.append(tlbr[3])
            xyxy.append(tlbr[2])

            return xyxy

        bbox_txt = open(bbox_txt_path, 'r')
        tlbr = []
        for line in bbox_txt:
            tlbr.append(int(line.split()[-1]))
        bbox = tlbr2xyxy(tlbr)
        bbox_txt.close()

        return bbox
    
    def parse_object_bbox_txt(self, object_bbox_txt_path):
        bboxes = []
        with open(object_bbox_txt_path, 'r') as bbox_txt:
            for line in bbox_txt:
                bboxes.append(list(map(float, line.split()))) # x, y, w, h, conf, class_
        
        return bboxes
        
    def parse_joints_txt(self, joint_txt_path):
        def map_joint_id_to_cmu_format(mhp_joints):
            joint_id_map = {
                17 : 0, 20 : 1,
                16 : 2, 18 : 3,19 : 4,
                1 : 5, 0 : 6, 2 : 7, 3 : 8,
                5 : 9, 4 : 10, 6 : 11, 7 : 12,
                13 : 13, 12 : 14, 14 : 15, 15 : 16,
                9 : 17, 8 : 18, 10 : 19, 11 : 20 
            }    # Note that the loc of mapped joint `1` has a different loc from common setting, so when define aggregate dataset, this joint should be masked
            # mapped_joints = np.zeros((len(joint_id_map.keys()), 2))
            mapped_joints = [(None, None) for _ in range(len(joint_id_map.keys()))]
            for mhp_id in joint_id_map.keys():
                cmu_id = joint_id_map[mhp_id]
                mapped_joints[cmu_id] = mhp_joints[mhp_id]
            return mapped_joints

        joint_txt = open(joint_txt_path, 'r')
        joints = []
        for line in joint_txt:
            name, x, y = line.split()
            joints.append((int(float(x)), int(float(y))))
        joint_txt.close()
        joints = map_joint_id_to_cmu_format(joints)

        return joints


if __name__ == '__main__':
    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/MHP_dataset'
    mhp_parser = MHPParser(data_root=data_root)
    print(len(mhp_parser.samples))

