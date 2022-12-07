import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from custom_layers import *

img_shape = (10, 1, 64,64)

def get_module_names(model):

    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names

def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, padding='zero', lrelu=True, batch_norm=False, w_norm=False, pixel_norm=False, only=False):
    if padding=='replication':
        layers.append(nn.ReplicationPad3d(pad))
        pad = 0
    if w_norm:  layers.append(EqualizedConv3d(c_in, c_out, k_size, stride, pad))
    else:   layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if lrelu:       layers.append(nn.LeakyReLU(0.2))
        else:           layers.append(nn.ReLU())
        if batch_norm:  layers.append(nn.BatchNorm3d(c_out))
        if pixel_norm:  layers.append(PixelwiseNormLayer())
    return layers

def conv(layers, c_in, c_out, k_size, stride=1, pad=0, padding='zero', lrelu=True, batch_norm=False, w_norm=False, d_gdrop=False, pixel_norm=False, only=False):

    if padding=='replication':
        layers.append(nn.ReplicationPad3d(pad))
        pad = 0
    if d_gdrop:         layers.append(GeneralizedDropOut(mode='prop', strength=0.0))
    if w_norm:          layers.append(EqualizedConv3d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:               layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if lrelu:       layers.append(nn.LeakyReLU(0.2))
        else:           layers.append(nn.ReLU())
        if batch_norm:  layers.append(nn.BatchNorm3d(c_out))
        if pixel_norm:  layers.append(PixelwiseNormLayer())
    return layers

def linear(layers, c_in, c_out, sig=True, w_norm=False):

    layers.append(Flatten())
    if w_norm:  layers.append(EqualizedLinear(c_in, c_out))
    else:       layers.append(nn.Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers

class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.nframes_pred = self.config.nframes_pred
        self.batch_norm = config.batch_norm
        self.w_norm = config.w_norm
        if self.config.loss=='lsgan':
            self.d_gdrop = True
        else:
            self.d_gdrop = config.d_gdrop
        self.padding = config.padding
        self.lrelu = config.lrelu
        self.d_sigmoid = config.d_sigmoid
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        if self.config.d_cond==False:
            self.nframes = self.config.nframes_pred
        else:
            self.nframes = config.nframes_in+config.nframes_pred
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def get_init_dis(self):

        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model

    def from_rgb_block(self, ndim):

        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        return  nn.Sequential(*layers)
    def last_block(self):

        # add MinibatchStdConcatLayer later.
        ndim = self.ndf
        layers = []
        layers.append(MinibatchStdConcatLayer())
        layers = conv(layers, ndim+1, ndim, 3, 1, 1, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        layers = conv(layers, ndim, ndim, (self.nframes,4,4), 1, 0, self.padding, self.lrelu, self.batch_norm, self.w_norm, self.d_gdrop, pixel_norm=False)
        layers = linear(layers, ndim, 1, sig=self.d_sigmoid, w_norm=self.w_norm)
        return  nn.Sequential(*layers), ndim

    def forward(self, x):
        x = self.model(x)
        return x

class AdversarialLoss(nn.Module):
    def __init__(self, gpu_id, args, gan_type='RGAN', gan_k=2,
                 lr_dis=1e-4):
        print('including adv loss')
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.batch_size = args.batch_size

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device('cuda:{}'.format(0))

        self.discriminator = Discriminator(args).to(self.device)

        self.optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )

        self.criterion_adv = GANLoss(gan_type='vanilla').to(self.device)

        self.real_label = torch.ones(self.batch_size)
        self.fake_label = torch.zeros(self.batch_size)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, fake, real):
        # D Loss
        for _ in range(self.gan_k):
            self.set_requires_grad(self.discriminator, True)
            self.optimizer.zero_grad()
            # real
            d_real = self.discriminator(real.detach())
            d_real_loss = self.criterion_adv(d_real, self.real_label)

            # fake
            d_fake = self.discriminator(fake.detach())
            d_fake_loss = self.criterion_adv(d_fake, self.fake_label)
            
            loss_d = d_real_loss + d_fake_loss
            
            loss_d.backward()
            self.optimizer.step()

        # G Loss
        self.set_requires_grad(self.discriminator, False)
        d_real = self.discriminator(real)
        loss_g = self.criterion_adv(d_real, self.real_label)

        # Generator loss
        return loss_g, loss_d

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict
