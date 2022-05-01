import os
import os.path as osp
import json
import shutil
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from transform import SimpleTransform

class HandPoseDataset(Dataset):
    def __init__(self, data_root, input_size=(256, 192), output_size=(64, 48), split='train'):
        # aka, model_input_size, model_output_size
        if split == 'train':
            self.data_dir = osp.join(data_root, 'manual_train')
        elif split == 'test':
            self.data_dir = osp.join(data_root, 'manual_test')
        else:
            print('unvalid split')
        self.collect_data()
        print("[INFO] generated [{}] data with [{}] samples.".format(split, len(self.jpg_path_set)))

        self.transform = SimpleTransform(input_size=input_size, output_size=output_size, train=(split=='train'))

    def collect_data(self):
        self.jpg_path_set, self.json_path_set = [], []
        for file_name in os.listdir(self.data_dir):
            assert(file_name.endswith('.jpg') or file_name.endswith('.json'))
            file_path = osp.join(self.data_dir, file_name)
            if file_name.endswith('.jpg'):
                self.jpg_path_set.append(file_path)
            else:
                self.json_path_set.append(file_path)
        self.jpg_path_set.sort()
        self.json_path_set.sort()
        # guarantee alignment
        assert(len(self.jpg_path_set) == len(self.json_path_set))
        for jpg_path, json_path in zip(self.jpg_path_set, self.json_path_set):
            assert(jpg_path.split('/')[-1].split('.')[0] == json_path.split('/')[-1].split('.')[0])

    def __getitem__(self, index):
        img = cv2.imread(self.jpg_path_set[index])
        with open(self.json_path_set[index], 'r') as label_file:
            label = json.load(label_file)
        
        img, target_mask, target_weight, bbox, joints = self.transform(img, label)
        # print(img.shape, target_mask.shape)

        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = torch.from_numpy(img).float()
        img /= 255
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)
        
        return img, target_mask, target_weight, bbox #, joints

    def __len__(self):
        return len(self.jpg_path_set)

if __name__ == '__main__':
    input_size = (256, 192)
    output_size = (64, 48)

    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/CMU_PANOPTIC/hand_labels'
    hand_pose_train_dataset = HandPoseDataset(data_root=data_root, input_size=input_size, output_size=output_size, split='train')
    hand_pose_test_dataset = HandPoseDataset(data_root=data_root, input_size=input_size, output_size=output_size, split='test')

    tmp_vis_dir = './tmp_vis'
    # if os.path.exists(tmp_vis_dir):
    #     shutil.rmtree(tmp_vis_dir)
    # os.mkdir(tmp_vis_dir)

    # for i in range(2590):
    #     print(i)
    i = 3
    print(hand_pose_train_dataset.jpg_path_set[i])
    print(hand_pose_train_dataset.json_path_set[i])
    full_img = cv2.imread(hand_pose_train_dataset.jpg_path_set[i])
    import json
    label = json.load(open(hand_pose_train_dataset.json_path_set[i]))
    orig_joints = label['hand_pts']

    sample = hand_pose_train_dataset.__getitem__(i)
    affined_img, target_mask, target_weight, bbox, joints = sample
    # print(img.shape, target_mask.shape, target_weight.shape)

    # clue1: even use the gt can not get right joint loc
    # maybe I got a wrong mask?
    from pose_utils.transforms import heatmap_to_coord_simple
    import copy
    preds, maxvals = heatmap_to_coord_simple(target_mask, bbox)
    for pt in preds:
        x, y = pt
        full_img = cv2.circle(copy.deepcopy(full_img), center=(int(x), int(y)), radius=2, color=(0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'recover_joint_on_full_img.jpg'), full_img)

    # clue2 : wrap to and wrap back can get the wright joint loc
    # from pose_utils.transforms import _box_to_center_scale, get_affine_transform, affine_transform, transform_preds
    # xmin, ymin, xmax, ymax = bbox
    # center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin)
    # height, width = input_size
    # trans = get_affine_transform(center, scale, rot=0, output_size=[width, height])
    # print("orig joints", orig_joints)
    # for j in range(len(joints)):
    #     joints[j][0:2] = affine_transform(joints[j][0:2], trans)
    # print("\ntrans joints", joints)
    # for j in range(len(joints)):
    #     joints[j][0:2] = transform_preds(np.array(joints[j][0:2]), center, scale, [width, height])   
    # print("\ntrans back joints", joints)

    # for pt in joints:
    #     x, y, _ = pt
    #     full_img = cv2.circle(full_img, center=(int(x), int(y)), radius=2, color=(0, 0, 255), thickness=1)
    # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'recover_joint_on_full_img.jpg'), full_img)


    # clue3: this can work
    # from pose_utils.transforms import get_heatmap_max_loc
    # coords, maxvals = get_heatmap_max_loc(target_mask)
    # # print("\nmax loc from mask", coords)
    # coords = 4 * np.array(coords)
    # # print("\n resize back max loc", coords)
    # from pose_utils.transforms import _box_to_center_scale, transform_preds
    # xmin, ymin, xmax, ymax = bbox
    # print("xmax - xmin, ymax - ymin", xmax - xmin, ymax - ymin)
    # center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin)
    # print("center, scale", center, scale)
    # height, width = input_size
    # for j in range(len(coords)):
    #     coords[j][0:2] = transform_preds(np.array(coords[j][0:2]), center, scale, [width, height]) 
    #     print(np.array(coords[j][0:2]), center, scale, [width, height]) 
    # for pt in coords:
    #     x, y = pt
    #     full_img = cv2.circle(full_img, center=(int(x), int(y)), radius=2, color=(0, 0, 255), thickness=1)
    # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'recover_joint_on_full_img.jpg'), full_img)








    # img = affined_img.numpy()
    # img = np.transpose(img, (1, 2, 0))   # C*H*W -> HWC
    # img *= 255
    # # target_mask = target_mask.numpy()
    # concat_target_mask = np.zeros((target_mask[0].shape))
    # for j in range(target_mask.shape[0]):
    #     concat_target_mask += target_mask[j]
    # concat_target_mask *= 255
    # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'img.jpg'), img)
    # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'mask.jpg'), concat_target_mask)

    # # # check mask with img
    # joint_loc = np.argwhere(concat_target_mask != 0)
    # # print(joint_loc)
    # for pt in joint_loc:
    #     y, x = pt
    #     img[y, x, :] = 255
    # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'affined_img_with_joints.jpg'), img)
    
    # # check joint id
    # # for j, pt in enumerate(joints):
    # #     x, y, _ = pt
    # #     cv2.putText(img, str(j),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    # # # # change output_size = 256, 192 and see here
    # # cv2.imwrite(os.path.join(tmp_vis_dir, str(i)+'img_with_joints.jpg'), img)

    # # recover geneated joints back to original img




    # # from torch.utils.data import DataLoader
    # # train_loader = DataLoader(hand_pose_train_dataset, batch_size=1,
    # #                         shuffle=True, num_workers=1)
    # # from tqdm import tqdm
    # # pbar = tqdm(total=len(train_loader))
    # # for i, (img, target_mask, target_weight) in enumerate(train_loader):
    # #     # print(i)
    # #     pbar.update()
