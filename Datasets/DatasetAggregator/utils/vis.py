import matplotlib.pyplot as plt
import cv2
import numpy as np

class PLTPoltter():
    def __init__(self, ax_size, **kwargs):
        self.fig, self.ax = plt.subplots(*ax_size, **kwargs)

    def plt_show_cv2_img(self, cv2_img, ax_id):
        self.ax[ax_id].imshow(cv2_img[:, :, ::-1])

    def plt_show_np_single_channel(self, np_array, channel, ax_id):
        array = np_array[channel]
        self.ax[ax_id].imshow(array)

    def plt_show_np_concat(self, np_array, concat_num, ax_id):
        channel, height, wdith = np_array.shape
        concat = np.zeros((height, wdith))
        for c in range(concat_num):
            # print(np.max(np_array[c]))
            concat += np_array[c]
        self.ax[ax_id].imshow(concat)

def draw_joints_on_img(img_, joints, show_joint_id=False, format='common'):
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
        [255, 0, 170], [255, 0, 85], [175, 80, 85], [25, 125, 100]
    ]

    img = img_.copy()
    for i, (pair, color) in enumerate(zip(pairs, colors)):
        ax, ay = joints[pair[0]]
        bx, by = joints[pair[1]]
        img = cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), color, thickness=2)
    
    for pt in joints:
        x, y = pt[0:2]
        cv2.circle(img, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=1)
        if show_joint_id:
            img = cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    return img

# def draw_bboxes_on_img(img_, bboxes):
#     img = img_.copy()
#     # bbox format: xywh
#     for bbox in bboxes:
#         x, y, w, h, conf, class_  = bbox
#         img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
#         img = cv2.putText(img, str(class_), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

#     return img