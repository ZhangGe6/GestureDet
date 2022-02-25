import torch
# from torch2trt import TRTModule
import cv2
import numpy as np
from model import UltraLightSimplePoseNet
from pose_utils.transforms import _box_to_center_scale, get_affine_transform, to_tensor, norm, heatmap_to_coord_simple, transform_preds, get_max_pred
from pose_utils.vis import draw_result_on_img


class UltraSimplePoseEstimator():
    def __init__(self, weight_path, trt_weight_path=None, trt_inf=False, process_img_size=(256, 192)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # (256, 192) is my trained img size
        if trt_inf:
            self.init_model_trt(trt_weight_path)
        else:
            self.init_model(weight_path)
        
        self.process_img_size = process_img_size  # height, width
        self.feat_stride = 4

    def init_model_trt(self, trt_weight_path):
        assert self.device == 'cuda'
        # load the saved model into a TRTModule
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_weight_path))
        self.model = model_trt

    def init_model(self, weight_path):
        self.model = UltraLightSimplePoseNet().eval().to(self.device)
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight)
    

    def get_pose(self, img, bboxes):
        '''
        height, width, _ = img.shape
        pred_joint_all_bboxes = []
        bbox_img = cv2.resize(img, dsize=(self.process_img_size[1], self.process_img_size[0]))
        w_ratio = self.process_img_size[1] / width
        h_ratio = self.process_img_size[0] / height

        bbox_img = to_tensor(bbox_img)
        bbox_img = norm(bbox_img, (-0.406, -0.457, -0.480))
        bbox_img = bbox_img.unsqueeze(0).to(self.device)

        heatmap = self.model(bbox_img)
        pred_joints_on_bbox, maxvals = get_max_pred(heatmap.squeeze(0).cpu().detach().numpy())
        pred_joints_on_img = [[joint[0] * self.feat_stride / w_ratio, joint[1] * self.feat_stride / h_ratio] for joint in pred_joints_on_bbox]

        pred_joint_all_bboxes.append((pred_joints_on_img, maxvals))
        print(pred_joint_all_bboxes)

        '''
        pred_joint_all_bboxes = []
        for bbox in bboxes:
            # bbox = bbox.cpu()
            # affined_img = self.pre_process_single_bbox(img, bbox).unsqueeze(0)
            xmin, ymin, xmax, ymax = map(int, bbox)
            bbox_img = img[ymin:ymax, xmin:xmax, :]
            bbox_img = cv2.resize(bbox_img, dsize=(self.process_img_size[1], self.process_img_size[0]))
            w_ratio = self.process_img_size[1] / (xmax - xmin)
            h_ratio = self.process_img_size[0] / (ymax - ymin)

            bbox_img = to_tensor(bbox_img)
            bbox_img = norm(bbox_img, (-0.406, -0.457, -0.480))
            bbox_img = bbox_img.unsqueeze(0).to(self.device)

            heatmap = self.model(bbox_img)
            pred_joints_on_bbox, maxvals = get_max_pred(heatmap.squeeze(0).cpu().detach().numpy())
            # pred_joints_on_bbox: [[x, y], [x, y], ...]
            pred_joints_on_img = [[joint[0] * self.feat_stride / w_ratio + xmin, joint[1] * self.feat_stride / h_ratio + ymin] for joint in pred_joints_on_bbox]

            # pred_joint_all_bboxes.append((pred_joints_on_img, maxvals))
            pred_joint_all_bboxes.append(pred_joints_on_img)
        
        return pred_joint_all_bboxes

    def draw_pose(self, img, poses):
        for pose in poses:
            img = draw_result_on_img(img, pose)

        return img


    

