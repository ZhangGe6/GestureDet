import cv2
import os
import time
import os.path as osp
import matplotlib.pyplot as plt

import sys
sys.path.append('./Detector/yolov5')
sys.path.append('./Detector/nanodet')
sys.path.append('./PoseEstimator/UltralightSimplePose')

from Detector.yolov5.yolov5_api import Yolov5Detector
from Detector.nanodet.nanodet_api import NanodetDetector
from PoseEstimator.UltralightSimplePose.UltraSimplePose_api import UltraSimplePoseEstimator
from Interactor.gesture_classifier import GestureClassifier
from Interactor.painter import GesturePainter

TRT_INF = True  # use TensorRT inference

detector = ['yolov5', 'nanodet'][1]
if detector == 'yolov5':
    weight_path = './Detector/yolov5/runs/train/exp2/weights/best.pt'
    detector = Yolov5Detector(weight_path=weight_path, process_img_size=640, yolo_thresh=0.5)

elif detector == 'nanodet':
    config_path = './Detector/nanodet/config/nanodet-m_hand.yml'
    weight_path = './Detector/nanodet/workspace/nanodet_m_hand//model_best/model_best.ckpt'
    trt_weight_path = './Detector/nanodet/workspace/nanodet_m_hand/deploy/tensorrt/nanodet_trt.pth'
    detector = NanodetDetector(config_path=config_path, weight_path=weight_path, trt_weight_path=trt_weight_path, conf_thresh=0.45, trt_inf=TRT_INF)

# pose_weight_path = './Ultralight-SimplePose/checkpoints/mobilenetv2_epoch_15_acc1_0.94.pt'
pose_weight_path = './PoseEstimator/UltralightSimplePose/checkpoints/mobilenetv2_epoch_40_acc1_0.96.pt'
trt_pose_weight_path = './PoseEstimator/UltralightSimplePose/deploy/tensorrt/hand_pose_trt.pth'
pose_estimator = UltraSimplePoseEstimator(weight_path=pose_weight_path, trt_weight_path=trt_pose_weight_path, trt_inf=TRT_INF)

gesture_classifer = GestureClassifier()
painter = GesturePainter()

cap = cv2.VideoCapture(0)
frame_id = 0
time_s = time.time()
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    detections = detector.get_detections(img)
    if detections.shape[0] > 0 and detections[detections[:, -1]==1].shape[0] > 0:  # there is at least one hand 
        hand_detections = detections[detections[:, -1]==1]
        bboxes = hand_detections[:, :4]
        hand_poses = pose_estimator.get_pose(img, bboxes)
        
        img = detector.detection_plot(img, hand_detections)
        img = pose_estimator.draw_pose(img, hand_poses)

        process_joints = hand_poses[0]  # support one hand currently
        gesture = gesture_classifer.pred_gesture(process_joints)

        fps = round((frame_id + 1) / (time.time() - time_s), 2)
        painter.update(img, gesture, process_joints, fps)
    else:
        painter.hang(img)
    
    # cv2.imwrite("./TMP_self/" + str(frame_id) + '.jpg', img)
    cv2.imshow('GestureDet-demo', img)
    cv2.waitKey(1)

    frame_id += 1
