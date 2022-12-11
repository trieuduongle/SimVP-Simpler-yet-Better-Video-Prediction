import torch
import torch.nn as nn
from model.modules import BaseModule

class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
        self, nc_in, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
        conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
            conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        # B, L, C, H, W = xs.shape
        # B, C, L, H, W = xs_t.shape
        xs_t = torch.transpose(xs, 1, 2)
        c1 = self.conv1(xs_t)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if self.use_sigmoid:
            c6 = torch.sigmoid(c6)
        out = torch.transpose(c6, 1, 2)
        return out
