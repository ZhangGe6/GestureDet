# This code is modified from https://github.com/skmhrk1209/QuanTorch/blob/HEAD/quantizers.py and
# https://github.com/666DZY666/micronet/blob/master/micronet/compression/quantization/wqaq/iao/quantize.py

import torch
from torch import nn
from torch import autograd

class Round(autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # sign = torch.sign(input)
        # output = sign * torch.floor(torch.abs(input) + 0.5)
        output = torch.round(input)
        # TODO the difference of using `torch.round` and `torch.floor(inp + 0.5)` ?
        return output

    @staticmethod
    def backward(ctx, grads):
        return grads

class Quantizer(nn.Module):
    
    def __init__(self, bit, observer, mode):
        super().__init__()
        self.bit = bit
        self.observer = observer
        self.register_quant_param_buffer()
        self.mode = mode
        assert(self.mode in ['training', 'calibrating', 'testing'])

    def register_quant_param_buffer(self):
        if self.observer.quant_granularity == 'layerwise':
            self.register_buffer('scale', torch.ones((1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((1), dtype=torch.float32))
        elif self.observer.quant_granularity == 'channelwise_conv':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.observer.quant_granularity == 'channelwise_fc':
            self.register_buffer('scale', torch.ones((self.observer.out_channels, 1), dtype=torch.float32))
            self.register_buffer('zero_point', torch.zeros((self.observer.out_channels, 1), dtype=torch.float32))
        self.register_buffer('eps', torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))
        self.buffer_init = True

    def update_params(self):
        raise NotImplementedError

    def round(self, input):
        outputs = Round.apply(input)
        return outputs

    def quantize(self, input):
        output = self.round(input / self.scale - self.zero_point)
        return output

    def clamp(self, input, min, max):
        outputs = torch.clamp(input, min, max)
        return outputs

    def dequantize(self, input):
        output = (input + self.zero_point) * self.scale
        return output

    def forward(self, input):
        if self.bit == 32:
            return input
        else:
            # if self.mode == 'training' or self.mode == 'calibrating':
            self.observer(input)   # update observer params (min max)
            self.update_params()   # update quantizer params (scale, zeropoint)
            # TODO: need clamp to (self.min, self.max) here? in case qat inf out of bound
            # if self.mode == 'testing':
            #     input = self.clamp(input, self.observer.min_val, self.observer.min_val)
            outputs = self.quantize(input)
            outputs = self.clamp(outputs, self.quant_min_val, self.quant_max_val)
            outputs = self.dequantize(outputs)
        return outputs


# see https://intellabs.github.io/distiller/algo_quantization.html for more info
class UnsignedQuantizer(Quantizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('quant_min_val', torch.tensor((0), dtype=torch.float32))
        self.register_buffer('quant_max_val', torch.tensor((1 << self.bit) - 1, dtype=torch.float32))
        # TODO: I do not know why the definitions of quant_min(quant max) weight and activation in micronet repo are different

class AsymmetricQuantizer(UnsignedQuantizer):
    
    def update_params(self):
        quantized_range = float(self.quant_max_val - self.quant_min_val)
        float_range = self.observer.max_val - self.observer.min_val
        self.scale = float_range / quantized_range
        self.scale = torch.max(self.scale, self.eps)    # processing for very small scale
        self.zero_point = torch.round(self.observer.min_val / self.scale)   
    

class SignedQuantizer(Quantizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('quant_min_val', torch.tensor(-(1 << (self.bit - 1)), dtype=torch.float32))
        self.register_buffer('quant_max_val', torch.tensor((1 << (self.bit - 1)) - 1, dtype=torch.float32))
        # TODO: I do not know why the definitions of quant_min(quant max) weight and activation in micronet repo are different
        # paper IAO 3.1 may be helpful

class SymmetricQuantizer(SignedQuantizer):
    
    # use full range as https://intellabs.github.io/distiller/algo_quantization.html illustates here
    def update_params(self):
        quantized_range = float(self.quant_max_val - self.quant_min_val) / 2    
        float_range = torch.max(torch.abs(self.observer.min_val), torch.abs(self.observer.max_val))
        self.scale = float_range / quantized_range
        self.scale = torch.max(self.scale, self.eps)  
        self.zero_point = torch.zeros_like(self.scale)



def get_quantizer(bit, observer, granularity, out_channels, is_symmetric, mode):
    if is_symmetric:
        quantizer = SymmetricQuantizer(bit=bit, 
                                       observer=observer(granularity=granularity, out_channels=out_channels),
                                       mode=mode)
    else:
        quantizer = AsymmetricQuantizer(bit=bit, 
                                       observer=observer(granularity=granularity, out_channels=out_channels),
                                       mode=mode)

    return quantizer