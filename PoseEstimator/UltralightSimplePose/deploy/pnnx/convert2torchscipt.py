import torch
import sys
sys.path.append('/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/PoseEstimator/UltralightSimplePose')
from model import UltraLightSimplePoseNet
from quantiation.utils import quantize_model

net = UltraLightSimplePoseNet().eval()
# ckpt = torch.load('/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/PoseEstimator/UltralightSimplePose/QAF_checkpoints/mobilenetv2_epoch_0_acc1_0.96.pt')
# net.load_state_dict(ckpt)
# net = quantize_model(net, args=())
ckpt = torch.load('/home/zg/wdir/zg/moyu/GestureDet/LightGestureDet/PoseEstimator/UltralightSimplePose/QAF_checkpoints/mobilenetv2_epoch_0_acc1_0.96.pt')
net.load_state_dict(ckpt, strict=False)

x = torch.rand(1, 3, 256, 192)
mod = torch.jit.trace(net, x)
torch.jit.save(mod, "./pose_torchscipt_qaf.pt")

# Then do: <path_to_pnnx>/pnnx ./pose_torchscipt.pt inputshape=[1,3,256,192] 
# as https://github.com/Tencent/ncnn/blob/master/tools/pnnx/README.md suggests