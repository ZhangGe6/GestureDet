import cv2
import os

image_folder = '../TMP_self'
video_name = '../video.avi'

frame = cv2.imread(os.path.join(image_folder, '0.jpg'))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 20, (width,height))
sample_rate = 5
# for frame_id in range(700, 810):
for frame_id in range(50, 327):
    video.write(cv2.imread(os.path.join(image_folder, str(frame_id) + '.jpg')))


cv2.destroyAllWindows()
video.release()