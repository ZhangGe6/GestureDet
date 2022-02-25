# refer to https://google.github.io/mediapipe/solutions/hands.html#resources

import os
import os.path as osp
import shutil
from tqdm import tqdm
import cv2
import json
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

data_root = './data'
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    for user_id in os.listdir(data_root):
        color_img_dir = osp.join(data_root, user_id, 'color')
        mediapipe_anno_dir = osp.join(data_root, user_id, 'mediapipe_anno')
        if osp.exists(mediapipe_anno_dir):
            print("{} already annodated skip".format(user_id))
            continue
        #     shutil.rmtree(mediapipe_anno_dir) # refresh if needed
        os.mkdir(mediapipe_anno_dir)

        print("labeling", user_id)
        for img_name in tqdm(os.listdir(color_img_dir)):
            if not img_name.endswith('.png'):
                continue
            img_path = osp.join(color_img_dir, img_name)
            # print(img_path)

            image = cv2.imread(img_path)
            height, width, _ = image.shape
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                continue
            
            res_dict = dict()
            res_dict['img_height'] = height
            res_dict['img_width'] = width
            res_dict['hands'] = []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):   # different hands
                hands_info = dict()
                hands_info['joints'] = []
                for land_mark in hand_landmarks.landmark:   # different joints
                    x, y, z = land_mark.x, land_mark.y, land_mark.z
                    unnormalize_x, unnormalize_y = x * width, y * height
                    hands_info['joints'].append((unnormalize_x, unnormalize_y))
            
                res_dict['hands'].append(hands_info)


            res_json_path = osp.join(mediapipe_anno_dir, img_name.split('.')[0] + '.json')
            with open(res_json_path, 'w') as f:
                json.dump(res_dict, f)

        #     break
        # break
        