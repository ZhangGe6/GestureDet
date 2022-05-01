import os
import json


val_json_path = '/home/zg/wdir/zg/moyu/GestureDet/Datasets/train_val_jsons/val_pose.json'
txt_path = './imagelist.txt'
with open(val_json_path, 'r') as f:
    val_json = json.load(f)
    samples = val_json['samples']

with open(txt_path, 'w') as f:
    for sample in samples:
        f.write(sample['image_path'] + '\n')


