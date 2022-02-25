import onnx
import onnxsim
import torch
import sys
sys.path.append('../..')
from model import UltraLightSimplePoseNet

net = UltraLightSimplePoseNet().eval()
ckpt = torch.load('../../checkpoints/mobilenetv2_epoch_15_acc1_0.94.pt')
net.load_state_dict(ckpt)

dummy_input = torch.autograd.Variable(
    torch.randn(1, 3, 256, 192)
)
output_path = './hand_pose.onnx'

torch.onnx.export(
    net,
    dummy_input,
    output_path,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=11,
    input_names=["data"],
    output_names=["output"],
)
print("finished exporting onnx ")
print("start simplifying onnx ")
input_data = {"data": dummy_input.detach().cpu().numpy()}
model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
if flag:
    onnx.save(model_sim, output_path)
    print("simplify onnx successfully")
else:
    print("simplify onnx failed")

# Then do: <path_to_pnnx>/pnnx ./pose_torchscipt.pt inputshape=[1,3,256,192] as 
# https://github.com/Tencent/ncnn/blob/master/tools/pnnx/README.md suggests