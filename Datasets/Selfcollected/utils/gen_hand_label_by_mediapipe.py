# refer to https://google.github.io/mediapipe/solutions/hands.html#resources

import os
import os.path as osp
import shutil
from tqdm import tqdm
import cv2
import json
import copy
import sys
# sys.path.append('../DatasetAggregator')
# from utils.vis import draw_joints_on_img

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

frames_dir = '../frames'
mediapipine_anno_path = '../hand_anno.json'
if os.path.exists(mediapipine_anno_path):  # refresh
    os.remove(mediapipine_anno_path)

res_dict = dict()
samples = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    for frame_name in tqdm(os.listdir(frames_dir)):
        sample = dict()

        frame_path = os.path.abspath(os.path.join(frames_dir, frame_name))
        sample['frame_path'] = frame_path

        image = cv2.imread(frame_path)
        height, width, _ = image.shape
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_hand_landmarks:
            print('no hand detected')
            sample['hands'] = []
            samples.append(sample)
            continue
        
        # img_with_joints = copy.deepcopy(image)
        hand_set = []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):   # different hands
            hand_joints = []
            for land_mark in hand_landmarks.landmark:   # different joints
                x, y, z = land_mark.x, land_mark.y, land_mark.z
                unnormalize_x, unnormalize_y = x * width, y * height
                hand_joints.append((unnormalize_x, unnormalize_y))

            hand_set.append(hand_joints)

        sample['hands'] = hand_set
        samples.append(sample)

        # break

assert len(samples) == len(os.listdir(frames_dir))
res_dict['info'] = "this is self collected data"
res_dict['samples'] = samples

with open(mediapipine_anno_path, 'a+') as anno:
    json.dump(res_dict, anno)



    