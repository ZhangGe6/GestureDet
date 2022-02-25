import os
import os.path as osp
import json
import shutil
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from transform import SimpleTransform
from pose_utils.transforms import to_tensor, norm

class HandPoseDataset(Dataset):
    def __init__(self, data_root, model_input_size=(256, 192), model_output_size=(64, 48), split='train', debug_vis=False):
        # aka, model_input_size, model_output_size
        json_path = osp.join(data_root, 'train_pose.json') if split == 'train' else osp.join(data_root, 'val_pose.json')
        self.collect_data(json_path)
        print("[INFO] generated [{}] samples.".format(len(self.samples)))
        self.debug_vis = debug_vis

        data_aug = (split=='train')
        self.transform = SimpleTransform(model_input_size=model_input_size, model_output_size=model_output_size, data_aug=data_aug)

    def collect_data(self, json_path):
        with open(json_path, 'r') as f:
            data_info = json.load(f)
            self.samples = data_info['samples']

    def __getitem__(self, index):
        img = cv2.imread(self.samples[index]['image_path'])
        bbox = self.samples[index]['hand_bbox']
        joints = self.samples[index]['joints']
        dataset_name = self.samples[index]['dataset_name']
        
        affined_img, target_mask, target_weight, affined_joints = self.transform(img, bbox, joints)
        if dataset_name == 'MHP': # mask the uncommon joint in MHP dataset
            target_weight[1][0][0] = 0
        # print(img.shape, target_mask.shape)
        if self.debug_vis:
            return affined_img, target_mask, target_weight, affined_joints, bbox, joints
        
        affined_img = to_tensor(affined_img)
        affined_img = norm(affined_img, (-0.406, -0.457, -0.480))
        
        return affined_img, target_mask, target_weight, affined_joints, self.samples[index]['image_path']

    def __len__(self):
        return len(self.samples)

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
    # #                         shuffle=False, num_workers=1)
    # # from tqdm import tqdm
    # # pbar = tqdm(total=len(train_loader))
    # # for i, (img, target_mask, target_weight) in enumerate(train_loader):
    # #     # print(i)
    # #     pbar.update()
