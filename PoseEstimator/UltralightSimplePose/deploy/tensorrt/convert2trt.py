import torch
from torch2trt import torch2trt

import sys
sys.path.append('/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/PoseEstimator/UltralightSimplePose')
from model import UltraLightSimplePoseNet

weight_path = '/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/PoseEstimator/UltralightSimplePose/checkpoints/mobilenetv2_epoch_40_acc1_0.96.pt'
model = UltraLightSimplePoseNet().cuda().eval()
weight = torch.load(weight_path)
model.load_state_dict(weight)

x = torch.ones((1, 3, 256, 192)).cuda()

# convert pytorch model to trt model (fp32) 
model_trt = torch2trt(model, [x])

# excute and check the output of the converted trt_model
y = model(x)
y_trt = model_trt(x)
print(torch.max(torch.abs(y - y_trt)))

# save the trt model as a state_dict.
torch.save(model_trt.state_dict(), './hand_pose_trt.pth')

