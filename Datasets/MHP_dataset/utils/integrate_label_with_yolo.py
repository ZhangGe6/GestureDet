import os
import shutil
import sys 
sys.path.append("/home/zg/wdir/zg/moyu/GestureDet/yolov5") 
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
# from utils.datasets import *
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device
from tqdm import tqdm


def cacu_IoU(box1_coor, box2_coor):
    box1_coor = box1_coor.cpu()
    box2_coor = box2_coor.cpu()
    box1_x1, box1_y1, box1_x2, box1_y2 = box1_coor[0], box1_coor[1], box1_coor[2], box1_coor[3]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)

    box2_x1, box2_y1, box2_x2, box2_y2 = box2_coor[0], box2_coor[1], box2_coor[2], box2_coor[3]
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    inter_w = min(box1_x2, box2_x2) - max(box1_x1, box2_x1)
    inter_h = min(box1_y2, box2_y2) - max(box1_y1, box2_y1)

    if inter_w <= 0 or inter_h <= 0:
        return 0

    inter_area = inter_w * inter_h
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def unnull_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def init_yolov5_model(weights, imgsz):
    # Initialize
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if half:
        model.half()  # to FP16
    
    return device, model, names, colors

def img_preprocess(img_path, imgsz, device):
    img0 = cv2.imread(img_path)  # BGR
    #print(img0.shape)
    img = letterbox(img0, imgsz)[0]
    #print(img.shape)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    #print(img.size())
    img = img.half() if device.type != 'cpu' else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img0, img

def get_detections(img0, img, model, names):
    conf_thres = 0.6
    iou_thres = 0.45
    classes = None
    agnostic_nms = False

    pred = model(img, augment=False)[0]
    # # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms) # type: list
    detections = pred[0]
    # detections: [
    # object1: [x1, y1, x2, y2, confidence, class]
    # object2: [x1, y1, x2, y2, confidence, class]
    # ...
    # ]
    detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()

    # s = ''
    # for c in detections[:, -1].unique():
    #     n = (detections[:, -1] == c).sum()  # detections per class
    #     s += '%g %ss, ' % (n, names[int(c)])  # add to string
    # print(s)

    return detections

def yolo2fall_label(detections, frame_action, action_area_coor):
    # person_num = 0
    # for detection in detections:
    #     if detection[5] == 0:
    #         person_num += 1
    # if (person_num > 1) and (torch.equal(action_area_coor, torch.zeros(4))):
    #     tkinter.messagebox.showinfo('info', 'Multiple person detected. Please specify the area where the action happens by drawing a rectangle')

    new_class_detection = []
    for detection in detections:
        # change all the initial detection that is not a person(initial label:0) to background(new label:4)
        # yolo label: ['person', 'bicycle', ... ]
        # fall label: ['stand', 'fall', 'squat', 'sit', 'background']
        # print('before changing: ', detection[5], names[int(detection[5])])
        if detection[5] == 0:
            # if person_num == 1: 
            #     detection[5] = frame_action
            # else:  # only set the person in action_area when multiple person are detected
            person_in_action_area = (detection[0] > action_area_coor[0] and detection[1] > action_area_coor[1] and detection[2] < action_area_coor[2]  and detection[3] < action_area_coor[3])
            if person_in_action_area:
                detection[5] = frame_action # if the person is in the action area, then it corresponds to the c3d label
            else:
                detection[5] = 0 # if the person is out of the action area, then is is regards as a default action 'stand'
            
        else:
            detection[5] = 4   # if not a person(yolo label), regard it as a background(fall label)
        # print('after changing: ', detection[5], new_names[int(detection[5])])
      
        new_class_detection.append(detection.cpu().numpy())

    new_class_detection = torch.tensor(new_class_detection)
    return new_class_detection
  
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def plot_and_label(img0, detections):
    global img1, video_name
    img1 = img0.copy() # init img1 each time the func is called. That means plot all of the current bbox on the basis of img0(an img without bbox)
    gn = torch.tensor(img1.shape)[[1, 0, 1, 0]] # normalization gain whwh
    for *xyxy, conf, cls in reversed(detections): # the labels of detections here have been changed to fall labels
        class_name = new_names[int(cls)]
        label = '%s %.2f' % (class_name, conf)
        plot_one_box(xyxy, img1, label=label, color=colors[int(cls)], line_thickness=3)
    # Then img1 is plotted with all the current bboxes and can be got in on_mouse(). 
    # cv2.imshow(video_name, img1)   

def save_label_txt(label_path, frame_id, detections):
    gn = torch.tensor(img1.shape)[[1, 0, 1, 0]] # normalization gain whwh

    with open(os.path.join(label_path, str(frame_id) + '.txt'), 'w+') as f:
        for *xyxy, conf, cls in reversed(detections):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 
            class_name = new_names[int(cls)]
            line = (int(cls.item()) , *xywh)  # label format

            f.write(('%s ' * len(line)).rstrip() % line + '\n')

def get_hand_locs_from_txt(txt_path, height, width):
    # to be [x1, y1, x2, y2, confidence, class]
    hands = []
    txt = open(txt_path)
    for line in txt:
        # refer to https://blog.csdn.net/u011520181/article/details/89218775
        hand_cls, norm_x_center, norm_y_center, norm_hand_width, norm_hand_height = map(float, line.split())
        x_center = norm_x_center * width
        y_center = norm_y_center * height
        hand_width = norm_hand_width * width
        hand_height = norm_hand_height * height
        
        x1 = x_center - hand_width / 2
        y1 = y_center - hand_height / 2
        x2 = x_center + hand_width / 2
        y2 = y_center + hand_height / 2
        
        hands.append([x1, y1, x2, y2, 1, 1])
    txt.close()
    
    return hands

def merge_dets(yolo_dets, predefined_hand):
    # to be [x1, y1, x2, y2, confidence, class]
    # cls 0 : person, 1 : hand, 2 : others 
    merged_dets = []
    for i, det in enumerate(yolo_dets):
        cls_ = det[-1]
        if cls_ >= 1:
            yolo_dets[i][-1] = 2  # force to be `others` class
        # now, 0 stands for person, and 1 is reserved for hand
        # TODO: whether overlap between hand and yolo dets need to be considered
        merged_dets.append(yolo_dets[i])
        
    merged_dets.append(predefined_hand)
    
    return merged_dets

def write_integrated_label(merged_dets, height, width, integrated_label_dir, label_txt_name):
    # src: [x1, y1, x2, y2, confidence, class]
    # dst: [norm_x_center, norm_y_center, norm_hand_width, norm_hand_height]
    with open(os.path.join(integrated_label_dir, label_txt_name), 'w+') as txt:
        for det in merged_dets:
            x1, y1, x2, y2, confidence, class_ = det
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            det_width = x2 - x1
            det_height = y2 - y1
            normed_x_center, normed_y_center = float(x_center) / width, float(y_center) / height
            normed_hand_width, normed_hand_height = float(det_width) / width, float(det_height) / height
            
            txt.write(str(int(class_)) + ' ' + str(normed_x_center) + ' ' + str(normed_y_center) + ' ' +
                        str(normed_hand_width) + ' ' + str(normed_hand_height) + '\n')

def parse_bbox_txt(bbox_txt_path):
    def tlbr2xyxy(tlbr):
        # tlbr: ymin, xmin, ymax, xmax
        # xyxy: xmin, ymin, xmax, ymax
        xyxy = []
        xyxy.append(tlbr[1])
        xyxy.append(tlbr[0])
        xyxy.append(tlbr[3])
        xyxy.append(tlbr[2])

        return xyxy

    bbox_txt = open(bbox_txt_path, 'r')
    tlbr = []
    for line in bbox_txt:
        tlbr.append(int(line.split()[-1]))
    bbox = tlbr2xyxy(tlbr)
    bbox_txt.close()

    return bbox

def add_post_fix(xyxy_bbox):
    
    return xyxy_bbox
     
def main():
    yolov5_path = '/home/zg/wdir/zg/moyu/GestureDet/yolov5/'
    weights = os.path.join(yolov5_path, 'yolov5x.pt')
    new_names = ['person', 'hand', 'others']
    imgsz = 640
    device, model, names, colors = init_yolov5_model(weights=weights, imgsz=imgsz)

    data_root_dir = '../'
    img_dir = os.path.join(data_root_dir, 'annotated_frames')
    hand_bbox_txt_dir = os.path.join(data_root_dir, 'bounding_boxes')
    objects_bbox_txt_dir = os.path.join(data_root_dir, 'objects_bounding_boxes')   # integrate hand and other objects
    if os.path.exists(objects_bbox_txt_dir):
        shutil.rmtree(objects_bbox_txt_dir)  # refresh
    os.mkdir(objects_bbox_txt_dir)

    for i, data_id in enumerate(os.listdir(img_dir)):
        print("collecting {}/{}: {} ...".format(i+1, len(os.listdir(img_dir)), data_id))
        os.mkdir(os.path.join(objects_bbox_txt_dir, data_id))

        for file_name in tqdm(os.listdir(os.path.join(img_dir, data_id))):
            if file_name.endswith('.jpg'):
                frame_id, camera_id = file_name.split('.jpg')[0].split('_webcam_')
                if camera_id == '4': continue  # bad label, skip
                
                hand_bbox_txt_name = frame_id + '_bbox_' + camera_id + '.txt'
                hand_bbox_txt_path = os.path.join(hand_bbox_txt_dir, data_id, hand_bbox_txt_name)
                assert(os.path.exists(hand_bbox_txt_path))
                hand_bbox = parse_bbox_txt(hand_bbox_txt_path)   # xyxy
                hand_bbox += [1, 1]                              # [x, y, x, y, confidence, class]           
            
                frame_path = os.path.join(img_dir, data_id, file_name)
                frame = cv2.imread(frame_path)
                # height, width, _ = frame.shape

                img0, img = img_preprocess(frame_path, imgsz, device)
                yolo_dets = get_detections(img0, img, model, names)   # [x, y, x, y, confidence, class]

                merged_dets = merge_dets(yolo_dets, hand_bbox)

                objects_bbox_txt_name = frame_id + '_objects_bbox_' + camera_id + '.txt'
                objects_bbox_txt_path = os.path.join(objects_bbox_txt_dir, data_id, objects_bbox_txt_name)

                # saved as xywh like COCO format
                with open(objects_bbox_txt_path, 'w+') as txt:
                    for det in merged_dets:
                        x1, y1, x2, y2, confidence, class_ = map(float, det)
                        w = x2 - x1
                        h = y2 - y1
                        txt.write(str(x1) + ' ' + str(y1) + ' ' +
                                  str(w) + ' ' + str(h) + ' ' + str(confidence) + ' ' + str(class_) + '\n')
                


if __name__ == "__main__":
    main()