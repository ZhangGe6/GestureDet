# This code is modified from https://github.com/666DZY666/micronet/blob/master/micronet/compression/quantization/wqaq/iao/quantize.py

import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self, granularity, out_channels=None):
        super(ObserverBase, self).__init__()
        self.quant_granularity = granularity
        self.out_channels = out_channels
        self.register_min_max_buffer()
    
    def register_min_max_buffer(self):
        if self.quant_granularity == 'layerwise':     # layer-wise
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        elif self.quant_granularity == 'channelwise_conv':   # channel-wise (for conv)
            self.register_buffer('min_val', torch.zeros((self.out_channels, 1, 1, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((self.out_channels, 1, 1, 1), dtype=torch.float32))
        elif self.quant_granularity == 'channelwise_fc':  # channel-wise (for fc)
            self.register_buffer('min_val', torch.zeros((self.out_channels, 1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((self.out_channels, 1), dtype=torch.float32))
        self.buffer_init = True
    
    def get_min_max(self, input):
        if self.quant_granularity == 'layerwise':     # layer-wise
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.quant_granularity == 'channelwise_conv':   # channel-wise (for conv)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.quant_granularity == 'channelwise_fc':  # channel-wise (for fc)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]
        
        return min_val, max_val

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        min_val, max_val = self.get_min_max(input)
        self.update_range(min_val, max_val)
    

class MinMaxObserver(ObserverBase):
    def __init__(self, granularity, out_channels=None):
        super(MinMaxObserver, self).__init__(granularity, out_channels)

    def update_range(self, min_val, max_val):
        if self.quant_granularity == 'channelwise_conv':
            min_val.resize_(self.min_val.shape)
            max_val.resize_(self.max_val.shape)

        if self.buffer_init:
            min_val = min_val
            max_val = max_val
            self.buffer_init = False
        else:
            min_val = torch.min(min_val, self.min_val)
            max_val = torch.max(max_val, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, granularity, out_channels=None, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(granularity, out_channels)
        self.momentum = momentum
    
    def update_range(self, min_val, max_val):
        if self.quant_granularity == 'channelwise_conv':
            min_val.resize_(self.min_val.shape)
            max_val.resize_(self.max_val.shape)

        if self.buffer_init:
            min_val = min_val
            max_val = max_val
            self.buffer_init = False
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

