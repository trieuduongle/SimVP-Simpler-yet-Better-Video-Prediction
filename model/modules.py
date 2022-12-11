import torch.nn as nn

from model.blocks import (
    GatedConv, GatedDeconv,
    PartialConv, PartialDeconv,
    VanillaConv, VanillaDeconv
)

class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv
            self.DeconvBlock = PartialDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv
