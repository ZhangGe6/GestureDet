import numpy as np
import cv2

class GesturePainter():
    def __init__(self, pen_color='red', pen_size=3, eraser_size=10):
        self.pen_bgr = {
            'red' : [0, 0, 255]
        }[pen_color]
        self.pen_size = pen_size
        self.eraser_size = eraser_size

        self.matte = None

        self.prev_gesture = None
    
    def init_matte(self, height, width):
        self.height, self.width = height, width
        self.matte = np.zeros((self.height, self.width))

    def update(self, img, gesture, joints, fps):
        if self.matte is None:
            height, width, _ = img.shape
            self.init_matte(height, width)

        if not gesture == self.prev_gesture:
            self.prev_pt = None
        self.prev_gesture = gesture

        if gesture == 'draw':
            self.change_matte(cur_pt=joints[4], value=1, size=self.pen_size)
        elif gesture == 'erase':
            self.change_matte(cur_pt=joints[8], value=0, size=self.eraser_size)
        elif gesture == 'reset':
            self.reset_matte()

        img[self.matte > 0] = self.pen_bgr
        img = cv2.putText(img, gesture, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        img = cv2.putText(img, str(fps) + ' fps', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (206, 135, 250), 2)

        return img

    def change_matte(self, cur_pt, value, size):
        if self.prev_pt is not None:
            pts = interp(self.prev_pt, cur_pt)
        else:
            pts = [cur_pt]
        self.prev_pt = cur_pt

        for pt in pts:
            x, y = map(int, pt)
            self.matte[bounded(y-size, 0, self.height-1):bounded(y+size, 0, self.height-1),\
                    bounded(x-size, 0, self.width-1):bounded(x+size, 0, self.width-1)] = value

    def reset_matte(self):
        self.matte = np.zeros((self.height, self.width))
    
    def hang(self, img):
        if self.matte is None:
            height, width, _ = img.shape
            self.init_matte(height, width)

        img[self.matte > 0] = self.pen_bgr
        img = cv2.putText(img, 'no hand', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        return img


    

def bounded(x, lower_bound, upper_bound):
    return min(max(lower_bound, x), upper_bound)

def interp(src_pt, dst_pt):
    src_x, src_y = map(int, src_pt)
    dst_x, dst_y = map(int, dst_pt)

    interp_pts = []
    x_gap = dst_x - src_x
    y_gap = dst_y - src_y

    steps = max(abs(x_gap), abs(y_gap))
    if steps == 0:
        return interp_pts
    x_step = x_gap / steps
    y_step = y_gap / steps
    for step in range(steps):
        interp_pts.append([int(src_x + x_step * step), int(src_y + y_step * step)])

    return interp_pts












   