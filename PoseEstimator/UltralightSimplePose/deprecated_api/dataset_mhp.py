import os
import os.path as osp
import json
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from transform import SimpleTransform
from pose_utils.parse_mhp import parse_bbox, parse_joints
from pose_utils.transforms import to_tensor, norm

class HandPoseDataset(Dataset):
    def __init__(self, data_root, model_input_size=(256, 192), model_output_size=(64, 48), split='train', debug_vis=False):
        # aka, model_input_size, model_output_size
        self.data_root = data_root
        self.frames_dir = osp.join(self.data_root, 'annotated_frames')
        self.bbox_txt_dir = osp.join(self.data_root, 'bounding_boxes')
        self.joint_txt_dir = osp.join(self.data_root, 'projections_2d')
        self.debug_vis = debug_vis
        self.split = split

        self.collect_data()
        print("[INFO] generated [{}] samples.".format(len(self.jpg_path_set)))

        self.transform = SimpleTransform(model_input_size=model_input_size, model_output_size=model_output_size, data_aug=(self.split=='train'))

    def collect_data(self):
        self.jpg_path_set, self.bbox_txt_path_set, self.joint_txt_path_set = [], [], []
        for data_id in tqdm(os.listdir(self.frames_dir)):
            for file_name in os.listdir(osp.join(self.frames_dir, data_id)):
                if file_name.endswith('.jpg'):
                    frame_id, camera_id = file_name.split('.jpg')[0].split('_webcam_')
                    if camera_id == '4':
                        continue
                    if self.split == 'train':
                        if int(frame_id) > 1500:
                            continue
                    if self.split == 'test':
                        if int(frame_id) <= 1500:
                            continue  

                    frame_path = osp.join(self.frames_dir, data_id, file_name)
                    self.jpg_path_set.append(frame_path)
                    
                    bbox_txt_name = frame_id + '_bbox_' + camera_id + '.txt'
                    bbox_txt_path = osp.join(self.bbox_txt_dir, data_id, bbox_txt_name)
                    assert(os.path.exists(bbox_txt_path))
                    self.bbox_txt_path_set.append(bbox_txt_path)

                    joint_txt_name = frame_id + '_jointsCam_' + camera_id + '.txt'
                    joint_txt_path = osp.join(self.joint_txt_dir, data_id, joint_txt_name)
                    assert(os.path.exists(joint_txt_path))
                    self.joint_txt_path_set.append(joint_txt_path)

    def __getitem__(self, index):
        img = cv2.imread(self.jpg_path_set[index])
        
        bbox = parse_bbox(self.bbox_txt_path_set[index])
        joints = parse_joints(self.joint_txt_path_set[index])
        
        affined_img, target_mask, target_weight, affined_joints = self.transform(img, bbox, joints)
        # print(img.shape, target_mask.shape)
        if self.debug_vis:
            return affined_img, target_mask, target_weight, affined_joints, bbox, joints
        
        affined_img = to_tensor(affined_img)
        affined_img = norm(affined_img, (-0.406, -0.457, -0.480))
        
        return affined_img, target_mask, target_weight, affined_joints#, bbox, joints

    def __len__(self):
        return len(self.jpg_path_set)


if __name__ == '__main__':
    model_input_size = (256, 192)
    mdoel_output_size = (64, 48)

    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/MHP_dataset'
    hand_pose_dataset = HandPoseDataset(data_root=data_root, model_input_size=model_input_size, model_output_size=mdoel_output_size)
    tmp_vis_dir = './tmp_vis'
    img, bbox, joints = hand_pose_dataset.__getitem__(0)

    ymin, xmin, ymax, xmax = bbox
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
    for pt in joints:
        x, y = pt
        img = cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=5)
    cv2.imwrite(os.path.join(tmp_vis_dir, 'vis_sample.jpg'), img)

    # # from torch.utils.data import DataLoader
    # # train_loader = DataLoader(hand_pose_train_dataset, batch_size=1,
    # #                         shuffle=True, num_workers=1)
    # # from tqdm import tqdm
    # # pbar = tqdm(total=len(train_loader))
    # # for i, (img, target_mask, target_weight) in enumerate(train_loader):
    # #     # print(i)
    # #     pbar.update()
