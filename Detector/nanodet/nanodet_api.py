# This file is largely refered to nanodet itself for fast implementation
# maybe slower than it can be
import os
import cv2
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# from torch2trt import TRTModule
import numpy as np

from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.model.head import NanoDetDummyHead

_COLORS = [(0, 0, 255), (255, 0, 0), (0, 0, 0)]
_CLASSES = ['person', 'hand', 'others']

class NanodetDetector():
    def __init__(self, config_path, weight_path, trt_weight_path=None, trt_inf=False, conf_thresh=0.45):
        load_config(cfg, config_path)
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if trt_inf:
            self.init_detector_trt(trt_weight_path)
        else:
            self.init_detector(weight_path)
        self.dummy_head = NanoDetDummyHead(**self.cfg.model.arch.head)

        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.conf_thresh = conf_thresh

    def init_detector_trt(self, trt_weight_path):
        assert self.device == 'cuda'
        # load the saved model into a TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_weight_path))
        self.model = model_trt

    def init_detector(self, weight_path):
        self.model = build_model(self.cfg.model).to(self.device).eval()
        ckpt = torch.load(weight_path)
        logger = Logger(-1, use_tensorboard=False)
        load_model_weight(self.model, ckpt, logger)
        
    def pre_process(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        return meta
        
    def post_process(self, results):
        all_box = []
        for label in results:
            for bbox in results[label]:
                score = bbox[-1]
                if score > self.conf_thresh:
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    all_box.append([x0, y0, x1, y1, score, label])

        return np.array(all_box)
        
    def get_detections(self, img):
        meta = self.pre_process(img)
    
        with torch.no_grad():
            preds = self.model(meta["img"])
            results = self.dummy_head.post_process(preds, meta)[0]

        results = self.post_process(results)

        return results

    def detection_plot(self, img, detections):

        for box in detections:
            x0, y0, x1, y1, score, label = box
            # color = self.cmap(i)[:3]
            color = _COLORS[int(label)]
            text = "{}:{:.1f}%".format(_CLASSES[int(label)], score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = 0.8
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)

            cv2.rectangle(
                img,
                (int(x0), int(y0 - txt_size - 1)),
                (int(x0 + txt_size + txt_size), int(y0 - 1)),
                color,
                -1,
            )
            cv2.putText(img, text, (int(x0), int(y0 - 1)), font, txt_size, color, thickness=1)
        return img




        





