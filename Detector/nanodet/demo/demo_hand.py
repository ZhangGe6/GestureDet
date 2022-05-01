import os
import cv2
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from nanodet.util import cfg, load_config, Logger
# config_path = '../config/nanodet-plus-m_320.xyml'
# model_path = '../workspace/nanodet-plus-m_320/nanodet-plus-m_320.pth'
# config_path = '../config/legacy_v0.x_configs/nanodet-m.yml'
# model_path = '../workspace/nanodet-m/nanodet-m.pth'
config_path = '../config/nanodet-plus-m_320_hand.yml'
model_path = '../workspace/nanodet-plus-m_320_hand/model_last.ckpt'

image_path = '/home/zg/wdir/zg/moyu/GestureDet/Datasets/MHP_dataset/annotated_frames/data_3/2_webcam_2.jpg'

load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)

from demo import Predictor

predictor = Predictor(cfg, model_path, logger, device=device)

meta, res = predictor.inference(image_path)
print(meta.keys())
# print(res[0])

# print(meta['raw_img'].shape)
print(len(meta['raw_img']))
print(res)
