import torch
import torch.nn as nn
import torch.nn.functional as F

from .observer import MinMaxObserver, MovingAverageMinMaxObserver
from .quantizer import get_quantizer

W_BIT = 8
A_BIT = 8

class QuantConv2d(nn.Conv2d):
    
    def __init__(self, *kargs, **kwargs):
        self.w_bit = kwargs.pop('w_bit', W_BIT)
        self.a_bit = kwargs.pop('a_bit', A_BIT)
        self.w_observer = kwargs.pop('w_observer', MinMaxObserver)
        self.a_observer = kwargs.pop('a_observer', MovingAverageMinMaxObserver)
        self.granularity = kwargs.pop('granularity', 'channelwise_conv')
        self.is_symmetric_w = kwargs.pop('is_symmetric_w', True)
        self.is_symmetric_a = kwargs.pop('is_symmetric_a', False)
        self.mode = kwargs.pop('mode', 'training')
        # print(self.w_bit, self.a_bit, self.w_observer, self.a_observer, self.granularity, self.is_symmetric_a, self.is_symmetric_w)

        super().__init__(*kargs, **kwargs)

        self.weight_quantizer = get_quantizer(bit=self.w_bit, 
                                            observer=self.w_observer,
                                            granularity=self.granularity,
                                            out_channels=self.out_channels,
                                            is_symmetric=self.is_symmetric_w,
                                            mode=self.mode
                                            )

        self.activation_quantizer = get_quantizer(bit=self.a_bit, 
                                            observer=self.a_observer,
                                            granularity='layerwise',
                                            out_channels=self.out_channels,
                                            is_symmetric=self.is_symmetric_a,
                                            mode=self.mode
                                            )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        
        return output
    
    # refer to torch/nn/qat/modules/conv
    @classmethod
    def from_float(cls, mod):
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                    groups=mod.groups, bias=mod.bias,
                    padding_mode=mod.padding_mode)
        qconv.weight = mod.weight
        qconv.bias = mod.bias

        return qconv
    
    def set_quant_mode(self, mode):
        # training / calibrating / testing
        self.activation_quantizer.mode = mode
        self.weight_quantizer.mode = mode


class QuantConvTranspose2d(nn.ConvTranspose2d):
    
    def __init__(self, *kargs, **kwargs):
        self.w_bit = kwargs.pop('w_bit', W_BIT)
        self.a_bit = kwargs.pop('a_bit', A_BIT)
        self.w_observer = kwargs.pop('w_observer', MinMaxObserver)
        self.a_observer = kwargs.pop('a_observer', MovingAverageMinMaxObserver)
        self.granularity = kwargs.pop('granularity', 'channelwise_conv')
        self.is_symmetric_w = kwargs.pop('is_symmetric_w', True)
        self.is_symmetric_a = kwargs.pop('is_symmetric_a', False)
        self.mode = kwargs.pop('mode', 'training')
        # print(self.w_bit, self.a_bit, self.w_observer, self.a_observer, self.granularity, self.is_symmetric_a, self.is_symmetric_w)

        super().__init__(*kargs, **kwargs)

        self.weight_quantizer = get_quantizer(bit=self.w_bit, 
                                            observer=self.w_observer,
                                            granularity=self.granularity,
                                            out_channels=self.out_channels,
                                            is_symmetric=self.is_symmetric_w,
                                            mode=self.mode
                                            )

        self.activation_quantizer = get_quantizer(bit=self.a_bit, 
                                            observer=self.a_observer,
                                            granularity='layerwise',
                                            out_channels=self.out_channels,
                                            is_symmetric=self.is_symmetric_a,
                                            mode=self.mode
                                            )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        quant_weight = self.weight_quantizer(self.weight)
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding, 
                                    self.groups, self.dilation)
        
        return output
    
    # refer to torch/nn/qat/modules/conv
    @classmethod
    def from_float(cls, mod):
        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                    stride=mod.stride, padding=mod.padding, output_padding=mod.output_padding, 
                    groups=mod.groups, bias=mod.bias, dilation=mod.dilation, padding_mode=mod.padding_mode
                    )
        qconv.weight = mod.weight
        qconv.bias = mod.bias

        return qconv
    
    def set_quant_mode(self, mode):
        # training / calibrating / testing
        self.activation_quantizer.mode = mode
        self.weight_quantizer.mode = mode