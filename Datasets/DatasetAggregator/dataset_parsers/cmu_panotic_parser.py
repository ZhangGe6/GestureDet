import os
import os.path as osp
import json
import cv2
from tqdm import tqdm
from utils.transforms import get_bbox_from_joints

class CMUPanoticParser():
    def __init__(self, data_root):
        self.data_root = data_root
        self.collect_samples()
        print("total {} vaild samples, {} skipped".format(len(self.samples), self.skipped_sample_num))

    def collect_samples(self):
        self.samples = []
        self.skipped_sample_num = 0
        for synth_split in os.listdir(self.data_root):
            if synth_split in ['synth1', 'synth4']: continue   # strange sample
            synth_split_dir = osp.join(self.data_root, synth_split)
            print('processing split {}...'.format(synth_split))
            for sample_name in tqdm(os.listdir(synth_split_dir)):
                if sample_name.endswith('.jpg'):
                    sample_img_name = sample_name
                    sample_json_name = sample_name.split('.jpg')[0] + '.json'

                    sample_img_path = os.path.abspath(osp.join(self.data_root, synth_split, sample_img_name))
                    sample_json_path = os.path.abspath(osp.join(self.data_root, synth_split, sample_json_name))
                    sample_img = cv2.imread(sample_img_path)
                    height, width, _ = sample_img.shape
                    sample_joints, sample_bbox = self.parse_json(sample_json_path, height, width)
                    # if sample_bbox is None:
                    #     self.skipped_sample_num += 1
                    #     continue
                    
                    sample = dict()
                    sample['img_path'] = sample_img_path
                    sample['img_size'] = (height, width)
                    sample['bbox'] = sample_bbox
                    sample['joints'] = sample_joints
                    self.samples.append(sample)
    
    def parse_json(self, json_path, height, width):
        with open(json_path, 'r') as label:
            label = json.load(label)
            hand_pts = []
            for pt in label['hand_pts']:
                x, y, _ = pt
                hand_pts.append([x, y])
            bbox = get_bbox_from_joints(hand_pts, height, width)
        return hand_pts, bbox
    
if __name__ == '__main__':
    data_root = '/home/zg/wdir/zg/moyu/GestureDet/datasets/CMU_PANOPTIC/hand_labels_synth'
    cmu_panotic_parser = CMUPanoticParser(data_root=data_root)
    print(len(cmu_panotic_parser.samples))

