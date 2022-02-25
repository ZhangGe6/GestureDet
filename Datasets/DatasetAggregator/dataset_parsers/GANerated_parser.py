import os
import os.path as osp
import json
import cv2
from tqdm import tqdm
from utils.transforms import get_bbox_from_joints

class GANeratedParser():
    def __init__(self, data_root):
        self.dataset_name = 'GANerated'
        self.data_root = data_root
        self.collect_samples()
        # print("total {} vaild samples, {} skipped".format(len(self.samples), self.skipped_sample_num))

    def collect_samples(self):
        self.samples = []
        self.skipped_sample_num = 0
        for user_id in os.listdir(self.data_root):
            print("collecting user {} ...".format(user_id))
            color_img_dir = osp.join(self.data_root, user_id, 'color')
            mediapipe_anno_dir = osp.join(self.data_root, user_id, 'mediapipe_anno')

            for img_name in tqdm(os.listdir(color_img_dir)):
                if not img_name.endswith('.png'):
                    continue
                img_path = osp.join(color_img_dir, img_name)
                json_name = img_name.split('.')[0] + '.json'
                json_path = osp.join(mediapipe_anno_dir, json_name)
                if not osp.exists(json_path):
                    # print("[Warning]: img {} can not find label json".format(img_name))
                    self.skipped_sample_num += 1
                    continue

                with open(json_path, 'r') as f:
                    label = json.load(f)

                    img_height, img_width = label['img_height'], label['img_width']
                    hand_joints = label['hands'][0]['joints']  # consider only 1 hand per img
                    bbox = get_bbox_from_joints(hand_joints, img_height, img_width)
                
                    sample = dict()
                    sample['dataset_name'] = self.dataset_name
                    sample['image_path'] = img_path
                    sample['img_size'] = (img_height, img_width)
                    sample['hand_bbox'] = bbox
                    sample['joints'] = hand_joints
                    self.samples.append(sample)
            # break
        

if __name__ == '__main__':
    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/RealHands/data'
    GANerated_parser = GANeratedParser(data_root=data_root)
    print(len(GANerated_parser.samples))

