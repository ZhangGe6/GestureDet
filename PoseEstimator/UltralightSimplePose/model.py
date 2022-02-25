# refer to https://github.com/dog-qiuqiu/Ultralight-SimplePose/blob/HEAD/posenet.py
# The original version is based on mxnet
import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

class MobileNetV2_backbone(torchvision.models.MobileNetV2):
    def __init__(self, *args, **kwargs):
        pretrained = kwargs.pop('pretrained', False)
        super().__init__(*args, **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['mobilenet_v2'])
            self.load_state_dict(state_dict)

        # refer to https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/43
        del self.classifier

    def forward(self, x):
        x = self.features(x)
        return x

class UltraLightSimplePoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # build backbone
        self.backbone = nn.Sequential(
            MobileNetV2_backbone(pretrained=True),
            nn.Conv2d(1280, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # build upsample module
        self.upsample_block1 = self.make_upsample_block()
        self.upsample_block2 = self.make_upsample_block()
        self.upsample_block3 = self.make_upsample_block()

        # bulid output conv
        self.out_conv = nn.Conv2d(128, 21, 1, stride=1, padding=0, bias=False)
        self._initialize()
    
    def make_upsample_block(self):
        return nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.upsample_block1(x)
        x = self.upsample_block2(x)
        x = self.upsample_block3(x)
        x = self.out_conv(x)

        return x
    
    def _initialize(self):
        for m in self.out_conv.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                # logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    net = UltraLightSimplePoseNet()
    # print(net)
    dummy_input = torch.rand(4, 3, 128, 64)
    dummy_output = net(dummy_input)
    print(dummy_output.shape)

