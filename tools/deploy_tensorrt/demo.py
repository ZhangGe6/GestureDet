import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model
model = alexnet(pretrained=True).eval().cuda()

# create a dummy input data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert pytorch model to trt model (fp32) 
model_trt = torch2trt(model, [x])

# excute and check the output of the converted trt_model
y = model(x)
y_trt = model_trt(x)
print(torch.max(torch.abs(y - y_trt)))

# save the trt model as a state_dict.
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

# load the saved model into a TRTModule
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('alexnet_trt.pth'))