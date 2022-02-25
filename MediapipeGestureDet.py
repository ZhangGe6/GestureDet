# refer to https://google.github.io/mediapipe/solutions/hands.html#resources
import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import cv2
import json
import copy
import sys
import time

sys.path.append('./PoseEstimator/UltralightSimplePose')
from pose_utils.vis import draw_result_on_img

from Interactor.gesture_classifier import GestureClassifier
from Interactor.painter import GesturePainter

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

gesture_classifer = GestureClassifier()
painter = GesturePainter()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    frame_id = 0
    time_s = time.time()
    while True:
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        height, width, _ = image.shape
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            det_hands = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):   # different hands
                hand_info = dict()
                hand_info['joints'] = []
                for land_mark in hand_landmarks.landmark:   # different joints
                    x, y, z = land_mark.x, land_mark.y, land_mark.z
                    unnormalize_x, unnormalize_y = x * width, y * height
                    hand_info['joints'].append([unnormalize_x, unnormalize_y])

                det_hands.append(hand_info)

            process_joints = det_hands[0]['joints']
            image = draw_result_on_img(image, process_joints, format='common')
            gesture = gesture_classifer.pred_gesture(process_joints)

            fps = round((frame_id + 1) / (time.time() - time_s), 2)
            painter.update(image, gesture, process_joints, fps)
        else:
            painter.hang(image)

        print(frame_id)
        cv2.imwrite("./TMP/" + str(frame_id) + '.jpg', image)

        cv2.imshow('GestureDet-Mediapipe', image)
        cv2.waitKey(1)
        frame_id += 1