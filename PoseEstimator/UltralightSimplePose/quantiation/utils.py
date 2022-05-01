# can refer to https://github.com/FLHonker/ZAQ-code/blob/main/quantization/quantize_model.py
import copy
from .quant_modules import *

def quantize_model(model, args):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """

    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = QuantConv2d.from_float(model)
        return quant_mod
    elif type(model) == nn.ConvTranspose2d:
        quant_mod = QuantConvTranspose2d.from_float(model)
        return quant_mod

    # recursively use the quantized module to replace the single-precision module
    # elif type(model) == nn.Sequential:
    # https://pytorch.org/vision/0.8/_modules/torchvision/models/mobilenet.html
    elif isinstance(model, nn.Sequential):   # for ConvBNReLU 
        # print()
        # print(model)
        mods = []
        for n, m in model.named_children():
            # print("sequential", n)
            mods.append(quantize_model(m, args))
        # print(mods)
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                setattr(q_model, attr, quantize_model(mod, args))
        return q_model


def set_quant_mode(model, mode):
    # quantize convolutional and linear layers to 8-bit
    if type(model) == QuantConv2d:
        model.set_quant_mode(mode)
        return model
    elif type(model) == nn.ConvTranspose2d:
        model.set_quant_mode(mode)
        return model

    # recursively change mode of modules
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(set_quant_mode(m, mode))
        return nn.Sequential(*mods)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(model, attr, set_quant_mode(mod, mode))
        return model
    
def freeze_weight_except_bias(model, verbose=False):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.requires_grad = False
    
    if verbose:    
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

