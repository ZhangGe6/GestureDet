import os
import cv2
import tqdm

video_src_dir = '../source_videos'
frames_dir = '../frames'
if not os.path.exists(frames_dir):
    os.mkdir(frames_dir)
sample_rate = 3

for video_name in os.listdir(video_src_dir):
    print("extracting video {} ...".format(video_name))
    cap = cv2.VideoCapture(os.path.join(video_src_dir, video_name))

    frame_id = 0
    ret, frame = cap.read()
    while ret:
        if frame_id % sample_rate == 0:
            frame_name = video_name.split('.mp4')[0] + '_' + str(frame_id) + '.jpg'
            cv2.imwrite(os.path.join(frames_dir, frame_name), frame)

        ret, frame = cap.read()
        frame_id += 1

print("Extracted {} frames from {} videos in total".format(len(os.listdir(frames_dir)), len(os.listdir(video_src_dir))))



