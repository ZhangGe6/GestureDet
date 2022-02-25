import cv2
import numpy as np
import torch
from .transforms import get_max_pred, norm

def joints_from_heatmap(heatmap_):
    if isinstance(heatmap_, np.ndarray):
        heatmap = heatmap_.copy()
    elif isinstance(heatmap_, torch.Tensor):
        heatmap = heatmap_.clone().detach()
        heatmap = heatmap.cpu().numpy()
    preds, maxvals = get_max_pred(heatmap)
    return preds, maxvals

def convert2_cv2_img(img_):
    if isinstance(img_, np.ndarray):
        img = img_.copy()
    elif isinstance(img_, torch.Tensor):
        img = img_.clone()
        img[0] += 0.406
        img[1] += 0.457
        img[2] += 0.480
        img = img * 255
        img = img.cpu().numpy()
        # img = np.ascontiguousarray(img, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        # img = np.clip(img, 0, 255)
    return img

def draw_joints(img_, joints, feat_stride=1):
    img = img_.copy()    # so buggy....
    for joint in joints:
        x, y = joint
        # Note 4 is the feature stride
        img = cv2.circle(img, center=(feat_stride * int(x), feat_stride * int(y)), radius=2, color=(255, 255, 255), thickness=2)
    return img

def draw_joints_pair(img, joints, feat_stride=1, format='common'):
    if format == 'MHP':
        pairs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[7,8],[1,9],[9,10],[10,11],[11,12],[1,13],[13,14],[14,15],[15,16],[1,17],[17,18],[18,19],[19,20]]
    else:
        pairs = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    colors = [
        [255, 0, 0], 
        [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
        [255, 0, 170], [255, 0, 85], [175, 80, 85], [25, 125, 100]]
    
    for pair, color in zip(pairs, colors):
        ax, ay = joints[pair[0]]
        bx, by = joints[pair[1]]
        img = cv2.line(img, (int(ax) * feat_stride, int(ay) * feat_stride), (int(bx) * feat_stride, int(by) * feat_stride), color, thickness=2)
    return img

def draw_result_from_heatmap(img_, heatmap, feat_stride=4, format='common'):
    preds, maxvals = joints_from_heatmap(heatmap)
    img = convert2_cv2_img(img_)
    img = draw_joints(img, preds, feat_stride=feat_stride)
    img = draw_joints_pair(img, preds, feat_stride=feat_stride, format=format)
    return img

def draw_result_on_img(img, joints, format='common'):
    img = draw_joints(img, joints)
    img = draw_joints_pair(img, joints, format=format)
    return img
